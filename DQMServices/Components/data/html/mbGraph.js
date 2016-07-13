var app = angular.module('mbGraph', []);

app.controller('InfoCtrl', function($scope, $http, $location, Profile, LocParams) {
    var me = this;

    me.fetch_info = function () {
        var p = $http({
            url: "mbGraph.json",
            method: 'GET',
        });

        p.then(function (b) {
            me.info = b.data;

            if (! LocParams.p.profile) {
                LocParams.p.profile = me.info.file;         
            }

            // try to find a (auto) reference
            var ib = me.info.env["CMSSW_GIT_HASH"];
            var arch = me.info.env["SCRAM_ARCH"];

            var find_prefix = function (p) {
                var re_search = [
                    new RegExp("\/DQMTestsResults\/DQMTestsResults\/([0-9a-zA-Z\_\.\/]*)\/mbGraph.html"),
                ];

                for (var i = 0; i < re_search.length; i++) {
                    var m = re_search[i].exec(p);
                    if (m) return m[1];
                }

                return null;
            };


            var prefix = find_prefix(window.location.pathname);
            if (prefix) {
                var ref_base = "/SDT/jenkins-artifacts/ib-dqm-tests/" + ib + "/" + arch + "/" + prefix;
                if (! LocParams.p.reference) {
                    LocParams.p.reference = ref_base + "/performance.json";
                }
            };
        });
    };

    me.fetch_info();
});

app.controller('GraphCtrl', function($scope, $http, $location, Profile, LocParams) {
    var me = this;

    me.set_profile = function () {
        var target = LocParams.p.profile;

        me.profile = null;
        me.profile_url = target;
        me.profile_error = null;

        if (! me.profile_url) {
            me.profile_error = "No profile provided";
            me.update_graph_data();
            return;
        }

        var p = Profile.load(target);
        p.then(function (body) {
            me.profile = body.data;
            me.update_graph_data();
        }, function (resp) {
            me.profile_error = "Failed to load profile: ";
            me.profile_error = me.reference_error + resp.status;
            me.update_graph_data();
        });
    };

    me.set_reference = function () {
        var target = LocParams.p.reference;
        if (!target) {
            return;
        }

        me.reference = null;
        me.reference_url = target;
        me.reference_error = null;
        var p = Profile.load(target);
        p.then(function (body) {
            me.reference = body.data;
            me.update_graph_data();
        }, function (resp) {
            me.reference_error = "Failed to load profile: ";
            me.reference_error = me.reference_error + resp.status;
            me.update_graph_data();
        });
    };

    me.update_graph_data = function () {
        me.graph_data = null;
        me.graph_data_reference = null;

        if (!me.profile) return;
        
        var pid = LocParams.p.pid;
        if (pid === undefined) {
            LocParams.setKey("pid", "_sum");
            return; // <- next digest will redraw
        }

        me.graph_data = me.profile[pid];

        if (!me.reference) return;

        if (pid == "_sum") {
            me.graph_data_reference = me.reference[pid];
        } else {
            me.graph_data_reference = null;
            _.some(me.reference, function (v) {
                if (v["cmdline"] == me.graph_data["cmdline"]) {
                    me.graph_data_reference = v;
                    return true;
                }
            });

        }
    };

    $scope.$watch(LocParams.watchFunc('pid'), me.update_graph_data);
    $scope.$watch(LocParams.watchFunc('profile'), me.set_profile);
    $scope.$watch(LocParams.watchFunc('reference'), me.set_reference);
});

app.service("Profile", ['$window', '$http', function($window, $http) {
    var x = {};
    var Profile = function () {
        var obj = {};
        return obj;
    };

    var pcnt2MB = function (x) {
        var PAGE = 4096;
        var MEGA = 1024*1024;
        return parseFloat(x) * PAGE / MEGA;
    };

    var parseFrame = function (profile, frame) {
        // do the sum magic (sum meta-process)
        if (! profile["_sum"]) {
            profile["_sum"] = {
                'start_ts': frame["time"],
                'frames': [],
                'cmdline': "sum of all processes",
                'cmdline_short': "sum of all processes"
            };
        };

        var sum_pdct = profile["_sum"];
        var sum_frame = {
            "p_run": 0, "p_stop": 0,
            "pss": 0, "rss": 0, "virt": 0,
            "time_diff": frame["time"] - sum_pdct["start_ts"]
        };

        // extract per pid memory usage and group it, well, per pid
        angular.forEach(frame["known_pids"], function (proc_dct, key) {
            // key of this dict is a pid
            if (!profile[key]) {
                profile[key] = {
                    'start_ts': frame["time"],
                    'frames': [],
                };
            }

            // update/fill the array
            pdct = profile[key];

            // this needs an update, in cast there was an exec
            pdct['cmdline'] = proc_dct["cmdline"];
            if (pdct['cmdline'].length > 42) {
                pdct["cmdline_short"] = pdct["cmdline"].substr(0, 45) + "...";
            } else {
                pdct["cmdline_short"] = pdct["cmdline"];
            }

            // parse statm
            var statm = proc_dct["statm"].split(" ");
            var virt = pcnt2MB(statm[0]);
            var rss = pcnt2MB(statm[1]);
            var pss = parseFloat(proc_dct["smaps_pss"]) / 1024 / 1024;

            // now fill in the sum_pdct
            if (proc_dct["running"]) {
                pdct["frames"].push({
                    'time_diff': frame["time"] - pdct["start_ts"],
                    'rss': rss,
                    'pss': pss,
                    'virt': virt,
                });

                sum_frame["p_run"] = sum_frame["p_run"] + 1;

                sum_frame["rss"]  = sum_frame["rss"]  + rss;
                sum_frame["pss"]  = sum_frame["pss"]  + pss;
                sum_frame["virt"] = sum_frame["virt"] + virt;
            } else {
                sum_frame["p_stop"] = sum_frame["p_stop"] + 1;
            }

            sum_pdct["frames"].push(sum_frame);
        });
    };

    var makeProfile = function (x) {
        var lines = x.split("\n");
        var frames = {};
        var profile = Profile();

        angular.forEach(lines, function (v) {
            var t = v.trim();
            if (t.length < 1)
                return;

            var parsed = JSON.parse(t.trim());
            parseFrame(profile, parsed);
        });

        return profile;
    };

    x.load = function (url) {
        var p = $http({
            url: url,
            method: 'GET',
            transformResponse: function(value, head, code) {
                if (code != 200) {
                    return null;
                };

                return makeProfile(value);
            }
        });

        return p;
    };

    return x;
}]);

app.directive('memoryGraph', function ($window) {
    var d3 = $window.d3;

    return {
        restrict: 'E',
        scope: { 'data': '=', 'width': '@', 'height': '@', 'referenceData': '=' },
        link: function (scope, elm, attrs) {
            var width = parseInt(scope.width);
            var height = parseInt(scope.height);

            var div = d3.select(elm[0]).append("div");
            div.attr("style", "position: relative");

            var svg = div.append("svg");
            svg.attr("width", width).attr("height", height);

            var chart = nv.models.lineChart()
                .margin({left: 100})
                .useInteractiveGuideline(false)
                .showLegend(true)
                .transitionDuration(350)
                .showYAxis(true)
                .showXAxis(true)
                .forceY(0)
            ;

            chart.interactiveLayer.tooltip.enabled(false);
            chart.interactiveLayer.tooltip.position({"left": 0, "top": 0});

            chart.xAxis
                .axisLabel('Time (s.)')
            //    .tickFormat(d3.time.format('%X'));
                .tickFormat(d3.format('.01f'));

            chart.yAxis
                .axisLabel('Mem (mb)')
                .tickFormat(d3.format('.02f'));

            var update = function () {
                var data = scope.data;

                //console.log("data", data);
                if (!data) {
                    svg.selectAll("*").remove();

                    svg
                        .datum([])
                        .transition().duration(500)
                        .call(chart)
                    return;
                }

                var make_xy = function (frames, metric, color, ref) {
                    var new_data = _.map(frames, function (frame) {
                        return { 'y': frame[metric], 'x': frame["time_diff"], '_obj': frame };
                    });

                    var key = metric;
                    if (ref) key = key + " (ref)";
                    return { values: new_data, 'key': key, 'color': color, classed: 'dashed' };
                };

                var datum = [
                    make_xy(data.frames, 'pss', "#1f77b4", false),
                    make_xy(data.frames, 'virt', "#ff7f0e", false),
                ];

                if (scope.referenceData) {
                    datum.push(make_xy(scope.referenceData.frames, 'pss', "#aec7e8", true));
                    datum.push(make_xy(scope.referenceData.frames, 'virt', "#ffbb78", true));
                }

                svg
                    .datum(datum)
                    .transition().duration(500)
                    .call(chart)
                ;
            };
            scope.$watch("data", update);
            scope.$watch("referenceData", update);
        }
    };
});

app.factory('LocParams', ['$location', '$rootScope', function ($location, $rootScope) {
    var me = {};

    me._value = function (v) {
        if (v === undefined) {
            return null;
        } else if (v === false) {
            return null;
        } else if (v === true) {
            return true;
        } else {
            return v;
        }
    };

    me._clear_object = function (obj) {
        for (var k in obj) {
            if (obj.hasOwnProperty(k))
                delete obj[k];
        }
    };

    // parameters inside a locaton (what we know)
    // cannot be modified by the scope
    me._params_location = {};

    // params inside the scope, we can modify this directly
    me._params = {};

    me._update_from_location  = function () {
        var s = $location.search();

        me._clear_object(me._params_location);
        me._clear_object(me._params);

        _.each(s, function (v, k) {
            me._params_location[k] = v;
            me._params[k] = v;
        });

        //console.log("params", me);
    };

    // change parameter with history
    me.setKey = function (k, v) {
        // this will propage to the _params on location event
        $location.search(k, me._value(v));
    };

    //// these are special "flags", they still modify the _params
    me.setFlag = function (flag_key, flag_char, value_bool) {
        var s = me._params[flag_key] || "";

        if ((value_bool) && (s.indexOf(flag_char) === -1))
            s += flag_char;

        if ((!value_bool) && (s.indexOf(flag_char) !== -1))
            s = s.replace(flag_char, '');

        me._params[flag_key] = s || null;
    };

    me.getFlag = function (flag_key, flag_char) {
        var s = me._params[flag_key] || "";
        return s.indexOf(flag_char) !== -1;
    };

    me.toggleFlag = function (flag_key, flag_char) {
        me.setFlag(flag_key, flag_char, !me.getFlag(flag_key, flag_char));
    };

    // short for function () { return LocParams.p.x; }
    me.watchFunc = function (key) {
        return function () { return me.p[key]; };
    };

    // watcher for async changer (history not advanced)
    $rootScope.$watch(function () { return me._params; }, function () {
        _.each(me._params, function (v, k) {
            var old = me._params_location[k];
            if (old !== v) {
                $location.search(k, me._value(v)).replace();
            };
        });
    }, true);

    $rootScope.$on("$locationChangeSuccess", me._update_from_location);
    me._update_from_location();

    me.p = me._params;
    $rootScope.LocParams = me;

    return me;
}]);
