app.directive('memoryGraph', function ($rootScope, $location, $window, LocParams) {
    var d3 = $window.d3;

    return {
        restrict: 'E',
        scope: {'data': '=', 'colors': '=', 'width': '@', 'height': '@'},
        templateUrl: "/chart.html",
        link: function (scope, elm, attrs) {

            // Breadcrumb dimensions: width, height, spacing, width of tip/tail.
            var b = {
                w: 155,
                h: 50,
                s: 10,
                t: 10
            };

            // Dimensions of sunburst frame
            var width = parseInt(scope.width);//window.innerWidth;
            var height = parseInt(scope.height); //window.innerHeight;
            var radius = Math.min(width, height) / 2;

            // Total size of all segments; we set this later, after loading the data.
            var totalSize = 0;

            var vis = d3.select("#chart").append("svg:svg")
                .attr("width", width / 2)
                .attr("x", 0)
                .attr("height", height)
                .attr("y", 0)
                .append("svg:g")
                .attr("id", "container")
                .attr("transform", "translate(" + width / 4 + "," + height / 2 + ")");

            var partition = d3.layout.partition()
                .size([2 * Math.PI, radius * radius])
                .value(function (d) {
                    return d.size;
                });

            var arc = d3.svg.arc()
                .startAngle(function (d) {
                    return d.x;
                })
                .endAngle(function (d) {
                    return d.x + d.dx;
                })
                .innerRadius(function (d) {
                    return Math.sqrt(d.y);
                })
                .outerRadius(function (d) {
                    return Math.sqrt(d.y + d.dy);
                });

            // Update after changed view mode.
            var update = function () {
                if (scope.data) {
                    vis.selectAll(".nodePath").remove();
                    d3.selectAll(".trail").remove();
                    showSunburst(scope.data, 1, scope.colors);
                }
            };

            function showSunburst(json, id, colors) {

                // Specific folder is selected by mouse click.
                var isClicked = false;

                createVisualization(json);

                // Main function to draw and set up the visualization, once we have the data.
                function createVisualization(json) {
                    // Basic setup of page elements.
                    if (id == 1) initializeBreadcrumbTrail();

                    // Bounding circle underneath the sunburst, to make it easier to detect
                    // when the mouse leaves the parent g.
                    vis.append("svg:circle")
                        .attr("r", radius)
                        .style("opacity", 0);

                    // For efficiency, filter nodes to keep only those large enough to see.
                    var nodes = partition.nodes(json)
                        .filter(function (d) {
                            return (d.dx > 0.005); // 0.005 radians = 0.29 degrees
                        });

                    var path = vis.data([json]).selectAll("path")
                        .data(nodes)
                        .enter().append("svg:path")
                        .classed("nodePath", true)
                        .attr("display", function (d) {
                            return d.depth ? null : "none";
                        })
                        .attr("d", arc)
                        .attr("fill-rule", "evenodd")
                        .style("fill", function (d) {
                            return colors[d.name];
                        })
                        .style("opacity", 1)
                        .on("mouseover", mouseover)
                        .on("click", mouseclick);

                    // Add the mouseleave handler to the bounding circle.
                    d3.select("#container").on("mouseleave", mouseleave);

                    // Get total size of the tree = value of root node from partition.
                    totalSize = path[0].parentNode.__data__.value;

                    // Read url parameter and set node as selected.
                    if (LocParams.p.selected) {
                        var path = LocParams.p.selected.split("-");
                        for (var i = 0; i < nodes.length; i++) {
                            if (path[0] === nodes[i].name) {
                                var k = 1, correct = true;
                                var current = nodes[i].parent;
                                while (current.parent) {
                                    if (path[k] != current.name) {
                                        correct = false;
                                        break;
                                    }
                                    current = current.parent;
                                    k++;
                                }
                                if (!correct) continue;
                                mouseover(nodes[i]);
                                isClicked = true;
                                break;
                            }
                        }
                    }
                };

                // Select or unset folder and change url parameter.
                function mouseclick(d) {
                    if (isClicked) {
                        isClicked = false;
                        LocParams.p.selected = null;
                        mouseover(d);
                    }
                    else {
                        isClicked = true;
                        var path = "";
                        path = d.name;
                        var current = d.parent;
                        while (current.parent) {
                            path += "-" + current.name;
                            current = current.parent;
                        }
                        LocParams.p.selected = path;
                    }
                    $rootScope.$apply();
                }

                // Fade all but the current sequence, and show it in the breadcrumb trail.
                function mouseover(d) {
                    if (isClicked) return;
                    var percentage = (100 * d.value / totalSize).toPrecision(3);
                    var percentageString = percentage + "%";
                    if (percentage < 0.05) {
                        percentageString = "< 0.05%";
                    }

                    d3.selectAll(".percentage")
                        .text(percentageString);

                    d3.select("#folder")
                        .text(d.name);

                    d3.selectAll(".size")
                        .text(d.value + "(B)");

                    d3.selectAll("#explanation")
                        .style("left", width / 4 - 100 / 2 + "px")
                        .style("top", (height - 100) / 2 + "px")
                        .style("visibility", "");

                    var sequenceArray = getAncestors(d);
                    updateBreadcrumbs(sequenceArray, percentageString);

                    // Fade all the segments.
                    d3.selectAll("path")
                        .style("opacity", 0.3);

                    // Then highlight only those that are an ancestor of the current segment.
                    d3.selectAll("path")
                        .filter(function (node) {
                            return (sequenceArray.indexOf(node) >= 0);
                        })
                        .style("opacity", 1);
                }

                // Restore everything to full opacity when moving off the visualization.
                function mouseleave(d) {
                    if (isClicked) return;
                    // Hide the breadcrumb trail
                    d3.select("#trail")
                        .style("visibility", "hidden");

                    // Deactivate all segments during transition.
                    d3.selectAll("path").on("mouseover", null);

                    // Transition each segment to full opacity and then reactivate it.
                    d3.selectAll("path")
                        .transition()
                        .duration(500)
                        .style("opacity", 1)
                        .each("end", function () {
                            d3.select(this).on("mouseover", mouseover);
                        });

                    d3.selectAll("#explanation")
                        .style("visibility", "hidden");
                }

                function getAncestors(node) {
                    var path = [];
                    var current = node;
                    while (current.parent) {
                        path.unshift(current);
                        current = current.parent;
                    }
                    return path;
                }

                function initializeBreadcrumbTrail() {

                    var trail = d3.select("#sequence")
                        .append("svg:svg")
                        .classed("trail", true)
                        .attr("width", window.innerWidth)
                        .attr("height", 50)
                        .attr("id", "trail");

                    // Add the label at the end, for the percentage.
                    trail.append("svg:text")
                        .attr("id", "endlabel")
                        .style("fill", "#000");
                };

                // Generate a string that describes the points of a breadcrumb polygon.
                function breadcrumbPoints(d, i) {
                    var points = [];
                    points.push("0,0");
                    points.push(b.w + ",0");
                    points.push(b.w + b.t + "," + (b.h / 2));
                    points.push(b.w + "," + b.h);
                    points.push("0," + b.h);
                    if (i > 0) { // Leftmost breadcrumb; don't include 6th vertex.
                        points.push(b.t + "," + (b.h / 2));
                    }
                    return points.join(" ");
                }

                // Update the breadcrumb trail to show the current sequence and percentage.
                function updateBreadcrumbs(nodeArray, percentageString) {

                    // Data join; key function combines name and depth (= position in sequence).
                    var g = d3.select("#trail")
                        .selectAll("g")
                        .data(nodeArray, function (d) {
                            return d.name + d.depth;
                        });

                    // Add breadcrumb and label for entering nodes.
                    var entering = g.enter().append("svg:g");

                    entering.append("svg:polygon")
                        .attr("points", breadcrumbPoints)
                        .style("fill", function (d) {
                            return colors[d.name];
                        });

                    entering.append("svg:text")
                        .attr("x", (b.w + b.t) / 2)
                        .attr("y", b.h / 2)
                        .attr("dy", "0.35em")
                        .attr("text-anchor", "middle")
                        .text(function (d) {
                            return d.name;
                        })
                        .call(wrap, 19);


                    function wrap(text, width) {
                        text.each(function () {
                            var text = d3.select(this),
                                words = text[0][0].__data__.name.match(new RegExp('.{1,' + width + '}', 'g')).reverse(),
                                word;
                            if (words.length > 2) text.attr("y", b.h / 4);
                            var line = [],
                                lineNumber = 0,
                                lineHeight = 1.1, // ems
                                x = text.attr("x"),
                                y = text.attr("y"),
                                dy = 0, //parseFloat(text.attr("dy")),
                                tspan = text.text(null)
                                    .append("tspan")
                                    .attr("x", x)
                                    .attr("y", y)
                                    .attr("dy", dy + "em");

                            var k = 0;
                            while (word = words.pop()) {
                                line.push(word);
                                tspan.text(line.join(" "));
                                if (tspan.node().getComputedTextLength() > b.w - b.t - b.t) {
                                    line.pop();
                                    tspan.text(line.join(" "));
                                    line = [word];
                                    if (!k) {
                                        tspan.text(word);
                                    }
                                    else {
                                        tspan = text.append("tspan")
                                            .attr("x", x)
                                            .attr("y", y)
                                            .attr("dy", ++lineNumber * lineHeight + dy + "em")
                                            .text(word);
                                    }

                                }
                                k++;
                            }
                        });
                    }

                    // Set position for entering and updating nodes.
                    g.attr("transform", function (d, i) {
                        return "translate(" + i * (b.w + b.s) + ", 0)";
                    });

                    // Remove exiting nodes.
                    g.exit().remove();

                    // Now move and update the percentage at the end.
                    d3.select("#trail").select("#endlabel")
                        .attr("x", (nodeArray.length + 0.5) * (b.w + b.s))
                        .attr("y", b.h / 2)
                        .attr("dy", "0.35em")
                        .attr("text-anchor", "middle")
                        .text(percentageString + " \n" + nodeArray[nodeArray.length - 1].value + "B");

                    // Make the breadcrumb trail visible, if it's hidden.
                    d3.select("#trail")
                        .style("visibility", "");
                }
            }

            scope.$watch("data", update);
        }

    };
});
