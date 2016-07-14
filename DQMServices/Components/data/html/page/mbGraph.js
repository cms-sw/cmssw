var app = angular.module('mbGraph', []);

app.controller('InfoCtrl', function ($scope, $http, $location, Profile, LocParams) {
    var me = this;

    me.fetch_info = function () {
        var p = $http({
            url: "mbGraph.json",
            method: 'GET',
        });

        p.then(function (b) {
            me.info = b.data;

            if (!LocParams.p.profile) {
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
                if (!LocParams.p.reference) {
                    LocParams.p.reference = ref_base + "/performance.json";
                }
            }
        });
    };

    me.fetch_info();
});

app.service('GraphService', function () {
    this.buildHierarchy = function (paths, totals, colors) {
        var root = {"name": "root", "children": []};
        for (var i = 0; i < paths.length; i++) {
            var size = parseInt(totals[i]);
            if (size <= 0) {
                continue;
            }
            var sequence = paths[i];
            var parts = sequence.split("\/");
            var currentNode = root;
            for (var j = 0; j < parts.length; j++) {
                var children = currentNode["children"];
                var nodeName = parts[j];
                colors[nodeName] = intToRGB(hashCode(nodeName));
                var childNode;
                if (j + 1 < parts.length) {
                    // Not yet at the end of the sequence; move down the tree.
                    var foundChild = false;
                    for (var k = 0; k < children.length; k++) {
                        if (children[k]["name"] == nodeName) {
                            childNode = children[k];
                            foundChild = true;
                            break;
                        }
                    }
                    // If we don't already have a child node for this branch, create it.
                    if (!foundChild) {
                        childNode = {"name": nodeName, "children": []};
                        children.push(childNode);
                    }
                    currentNode = childNode;
                } else {
                    // Reached the end of the sequence; create a leaf node.
                    childNode = {"name": nodeName, "size": size};
                    children.push(childNode);
                }
            }
        }
        return root;
    };

    function hashCode(str) {
        var hash = 0;
        for (var i = 0; i < str.length; i++) {
            hash = str.charCodeAt(i) + ((hash << 5) - hash);
        }
        return hash;
    }

    function intToRGB(i) {
        var c = (i & 0x00FFFFFF)
            .toString(16)
            .toUpperCase();
        return "#" + "00000".substring(0, 6 - c.length) + c;
    }

    function getIndex(array, key) {
        var lo = 0,
            hi = array.length - 1,
            mid,
            element;
        while (lo <= hi) {
            mid = ((lo + hi) >> 1);
            element = array[mid];
            if (element < key) {
                lo = mid + 1;
            } else if (element > key) {
                hi = mid - 1;
            } else {
                return mid;
            }
        }
        return -1;
    }

    this.compare = function (paths1, totals1, paths2, totals2) {
        for (var indA = 0; indA < paths1.length; indA++) {
            var valA = parseInt(totals1[indA]);
            if (valA <= 0) {
                continue;
            }
            var item = paths1[indA];
            var indB = getIndex(paths2, item);
            if (indB >= 0) {
                var valB = totals2[indB];
                var diff = valA - valB;
                if (diff == 0) {
                    totals1[indA] = -1;
                    totals2[indB] = -1;
                }
                if (diff > 0) {
                    totals1[indA] = diff;
                    totals2[indB] = -1;
                }
                if (diff < 0) {
                    totals2[indB] = Math.abs(diff);
                    totals1[indA] = -1;
                }
            }
        }
    }

})

app.controller('GraphCtrl', function ($scope, $http, $location, Profile, LocParams, GraphService) {
    var me = this;

    me.set_profile = function () {
        var target = LocParams.p.profile;
        if (!target) return;

        me.profile = null;
        me.profile_url = target;
        me.profile_error = null;

        if (!me.profile_url) {
            me.profile_error = "No profile provided";
            me.update_graph_data();
            return;
        }

        var p = Profile.load(target);
        p.then(function (response) {
            me.profile = {};
            me.profile.paths = angular.fromJson(response.data.histograms[0].path);
            me.profile.totals = angular.fromJson(response.data.histograms[0].total);
            me.profile.colors = {};

            me.diffProfile = {};
            me.diffProfile.paths = me.profile.paths.slice(0);
            me.diffProfile.totals = me.profile.totals.slice(0);
            me.diffProfile.colors = {};

            me.list = ["profile"];
            me.update_graph_data();
        }, function (resp) {
            me.profile_error = "Failed to load profile: ";
            me.profile_error = me.reference_error + resp.status;
            me.update_graph_data();
        });
    };

    me.set_reference = function () {
        var target = LocParams.p.reference;
        if (!target) return;

        me.reference = null;
        me.reference_url = target;
        me.reference_error = null;

        var p = Profile.load(target);
        p.then(function (response) {
            me.reference = {};
            me.reference.paths = angular.fromJson(response.data.histograms[0].path);
            me.reference.totals = angular.fromJson(response.data.histograms[0].total);
            me.reference.colors = {};

            me.diffRef = {};
            me.diffRef.paths = me.reference.paths.slice(0);
            me.diffRef.totals = me.reference.totals.slice(0);
            me.diffRef.colors = {};

            me.list.push("reference");
            me.list.push("difference profile");
            me.list.push("difference reference");
            me.update_graph_data();
        }, function (resp) {
            me.reference_error = "Failed to load profile: ";
            me.reference_error = me.reference_error + resp.status;
            me.update_graph_data();
        });
    };

    me.update_graph_data = function () {
        if (!LocParams.p.show) LocParams.p.show = "profile";
        me.graph_data = null;
        if (!me.profile) return;

        if (!me.reference) {
            me.graph_data = GraphService.buildHierarchy(me.profile.paths, me.profile.totals, me.profile.colors);
            me.colors = me.profile.colors;
            return;
        }
        switch (LocParams.p.show) {
            case "profile":
                me.graph_data = GraphService.buildHierarchy(me.profile.paths, me.profile.totals, me.profile.colors);
                me.colors = me.profile.colors;
                break;
            case "reference":
                me.graph_data = GraphService.buildHierarchy(me.reference.paths, me.reference.totals, me.reference.colors);
                me.colors = me.reference.colors;
                break;
            case "difference profile":
                GraphService.compare(me.diffProfile.paths, me.diffProfile.totals, me.diffRef.paths, me.diffRef.totals);
                me.diffProfile.colors = {};
                me.graph_data = GraphService.buildHierarchy(me.diffProfile.paths, me.diffProfile.totals, me.diffProfile.colors);
                me.colors = me.diffProfile.colors;
                break;
            case "difference reference":
                GraphService.compare(me.diffProfile.paths, me.diffProfile.totals, me.diffRef.paths, me.diffRef.totals);
                me.diffRef.colors = {};
                me.graph_data = GraphService.buildHierarchy(me.diffRef.paths, me.diffRef.totals, me.diffRef.colors);
                me.colors = me.diffRef.colors;
                break;
            default:
                me.graph_data = GraphService.buildHierarchy(me.profile.paths, me.profile.totals, me.profile.colors);
                me.colors = me.profile.colors;
                break;
        }
    };

    $scope.$watch(LocParams.watchFunc('show'), me.update_graph_data);
    $scope.$watch(LocParams.watchFunc('profile'), me.set_profile);
    $scope.$watch(LocParams.watchFunc('reference'), me.set_reference);
});

app.service("Profile", ['$window', '$http', function ($window, $http) {
    var x = {};
    var Profile = function () {
        var obj = {};
        return obj;
    };

    x.load = function (url) {
        var p = $http({
            url: url,
            method: 'GET'
        });
        return p;
    };

    return x;
}]);

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

    // parameters inside a location (what we know)
    // cannot be modified by the scope
    me._params_location = {};

    // params inside the scope, we can modify this directly
    me._params = {};

    me._update_from_location = function () {
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
        return function () {
            return me.p[key];
        };
    };

    // watcher for async changer (history not advanced)
    $rootScope.$watch(function () {
        return me._params;
    }, function () {
        _.each(me._params, function (v, k) {
            var old = me._params_location[k];
            if (old !== v) {
                $location.search(k, me._value(v)).replace();
            }
        });
    }, true);

    $rootScope.$on("$locationChangeSuccess", me._update_from_location);
    me._update_from_location();

    me.p = me._params;
    $rootScope.LocParams = me;

    return me;
}]);