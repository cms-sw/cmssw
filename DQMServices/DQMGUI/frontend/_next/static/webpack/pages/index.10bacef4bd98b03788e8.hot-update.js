webpackHotUpdate_N_E("pages/index",{

/***/ "./components/utils.ts":
/*!*****************************!*\
  !*** ./components/utils.ts ***!
  \*****************************/
/*! exports provided: seperateRunAndLumiInSearch, get_label, getPathName, makeid, getZoomedPlotsUrlForOverlayingPlotsWithDifferentNames, getZoomedOverlaidPlotsUrlForOverlayingPlotsWithDifferentNames, decodePlotName */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "seperateRunAndLumiInSearch", function() { return seperateRunAndLumiInSearch; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_label", function() { return get_label; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getPathName", function() { return getPathName; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "makeid", function() { return makeid; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getZoomedPlotsUrlForOverlayingPlotsWithDifferentNames", function() { return getZoomedPlotsUrlForOverlayingPlotsWithDifferentNames; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getZoomedOverlaidPlotsUrlForOverlayingPlotsWithDifferentNames", function() { return getZoomedOverlaidPlotsUrlForOverlayingPlotsWithDifferentNames; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "decodePlotName", function() { return decodePlotName; });
var seperateRunAndLumiInSearch = function seperateRunAndLumiInSearch(runAndLumi) {
  var runAndLumiArray = runAndLumi.split(':');
  var parsedRun = runAndLumiArray[0];
  var parsedLumi = runAndLumiArray[1] ? parseInt(runAndLumiArray[1]) : 0;
  return {
    parsedRun: parsedRun,
    parsedLumi: parsedLumi
  };
};
var get_label = function get_label(info, data) {
  var value = data ? data.fString : null;

  if (info !== null && info !== void 0 && info.type && info.type === 'time' && value) {
    var milisec = new Date(parseInt(value) * 1000);
    var time = milisec.toUTCString();
    return time;
  } else {
    return value ? value : 'No information';
  }
};
var getPathName = function getPathName() {
  var isBrowser = function isBrowser() {
    return true;
  };

  var pathName = isBrowser() && window.location.pathname || '/';
  var the_lats_char = pathName.charAt(pathName.length - 1);

  if (the_lats_char !== '/') {
    pathName = pathName + '/';
  }

  return pathName;
};
var makeid = function makeid() {
  var text = '';
  var possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';

  for (var i = 0; i < 5; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }

  return text;
};
var getZoomedPlotsUrlForOverlayingPlotsWithDifferentNames = function getZoomedPlotsUrlForOverlayingPlotsWithDifferentNames(query, selected_plot) {
  var page = 'plotsLocalOverlay';
  var run = 'run_number=' + query.run_number;
  var dataset = 'dataset_name=' + query.dataset_name;
  var path = 'folders_path=' + selected_plot.path;
  var plot_name = 'plot_name=' + selected_plot.name;
  var queryURL = [run, dataset, path, plot_name].join('&');
  var plotsLocalOverlayURL = [page, queryURL].join('?');
  return plotsLocalOverlayURL;
};
var getZoomedOverlaidPlotsUrlForOverlayingPlotsWithDifferentNames = function getZoomedOverlaidPlotsUrlForOverlayingPlotsWithDifferentNames(query, selected_plot) {
  var _query$overlay_data;

  var page = 'plotsLocalOverlay';
  var run = 'run_number=' + query.run_number;
  var dataset = 'dataset_name=' + query.dataset_name;
  var path = 'folders_path=' + selected_plot.path;
  var plot_name = 'plot_name=' + selected_plot.name;
  var globally_overlaid_plots = (_query$overlay_data = query.overlay_data) === null || _query$overlay_data === void 0 ? void 0 : _query$overlay_data.split('&').map(function (plot) {
    var parts = plot.split('/');
    var run_number = parts.shift();
    var pathAndLabel = parts.splice(3);
    var dataset_name = parts.join('/');
    var path = selected_plot.path;
    var plot_name = selected_plot.name;
    var label = pathAndLabel.pop();
    var string = [run_number, dataset_name, path, plot_name, label].join('/');
    return string;
  });
  var global_overlay = 'overlaidGlobally=' + globally_overlaid_plots.join('&');
  var queryURL = [run, dataset, path, plot_name, global_overlay].join('&');
  var plotsLocalOverlayURL = [page, queryURL].join('?');
  return plotsLocalOverlayURL;
};
var decodePlotName = function decodePlotName(tooLong, plot_name) {
  if (tooLong) {
    var decode_name = decodeURI(plot_name);
    return decode_name.substring(0, 25) + '...'; //some of names are double encoded 
  } else {
    return decodeURI(plot_name);
  }
};

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy91dGlscy50cyJdLCJuYW1lcyI6WyJzZXBlcmF0ZVJ1bkFuZEx1bWlJblNlYXJjaCIsInJ1bkFuZEx1bWkiLCJydW5BbmRMdW1pQXJyYXkiLCJzcGxpdCIsInBhcnNlZFJ1biIsInBhcnNlZEx1bWkiLCJwYXJzZUludCIsImdldF9sYWJlbCIsImluZm8iLCJkYXRhIiwidmFsdWUiLCJmU3RyaW5nIiwidHlwZSIsIm1pbGlzZWMiLCJEYXRlIiwidGltZSIsInRvVVRDU3RyaW5nIiwiZ2V0UGF0aE5hbWUiLCJpc0Jyb3dzZXIiLCJwYXRoTmFtZSIsIndpbmRvdyIsImxvY2F0aW9uIiwicGF0aG5hbWUiLCJ0aGVfbGF0c19jaGFyIiwiY2hhckF0IiwibGVuZ3RoIiwibWFrZWlkIiwidGV4dCIsInBvc3NpYmxlIiwiaSIsIk1hdGgiLCJmbG9vciIsInJhbmRvbSIsImdldFpvb21lZFBsb3RzVXJsRm9yT3ZlcmxheWluZ1Bsb3RzV2l0aERpZmZlcmVudE5hbWVzIiwicXVlcnkiLCJzZWxlY3RlZF9wbG90IiwicGFnZSIsInJ1biIsInJ1bl9udW1iZXIiLCJkYXRhc2V0IiwiZGF0YXNldF9uYW1lIiwicGF0aCIsInBsb3RfbmFtZSIsIm5hbWUiLCJxdWVyeVVSTCIsImpvaW4iLCJwbG90c0xvY2FsT3ZlcmxheVVSTCIsImdldFpvb21lZE92ZXJsYWlkUGxvdHNVcmxGb3JPdmVybGF5aW5nUGxvdHNXaXRoRGlmZmVyZW50TmFtZXMiLCJnbG9iYWxseV9vdmVybGFpZF9wbG90cyIsIm92ZXJsYXlfZGF0YSIsIm1hcCIsInBsb3QiLCJwYXJ0cyIsInNoaWZ0IiwicGF0aEFuZExhYmVsIiwic3BsaWNlIiwibGFiZWwiLCJwb3AiLCJzdHJpbmciLCJnbG9iYWxfb3ZlcmxheSIsImRlY29kZVBsb3ROYW1lIiwidG9vTG9uZyIsImRlY29kZV9uYW1lIiwiZGVjb2RlVVJJIiwic3Vic3RyaW5nIl0sIm1hcHBpbmdzIjoiOzs7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QUFiLEVBQTJCQyxJQUEzQixFQUFpQ0MsU0FBakMsRUFBNENjLEtBQTVDLEVBQW1EWCxJQUFuRCxDQUF3RCxHQUF4RCxDQUFmO0FBQ0EsV0FBT2EsTUFBUDtBQUNELEdBVitCLENBQWhDO0FBV0EsTUFBTUMsY0FBYyxHQUFHLHNCQUF1QlgsdUJBQUQsQ0FBc0NILElBQXRDLENBQTJDLEdBQTNDLENBQTdDO0FBQ0EsTUFBTUQsUUFBUSxHQUFHLENBQUNQLEdBQUQsRUFBTUUsT0FBTixFQUFlRSxJQUFmLEVBQXFCQyxTQUFyQixFQUFnQ2lCLGNBQWhDLEVBQWdEZCxJQUFoRCxDQUFxRCxHQUFyRCxDQUFqQjtBQUNBLE1BQU1DLG9CQUFvQixHQUFHLENBQUNWLElBQUQsRUFBT1EsUUFBUCxFQUFpQkMsSUFBakIsQ0FBc0IsR0FBdEIsQ0FBN0I7QUFDQSxTQUFPQyxvQkFBUDtBQUNELENBckJNO0FBd0JBLElBQU1jLGNBQWMsR0FBRyxTQUFqQkEsY0FBaUIsQ0FBQ0MsT0FBRCxFQUFtQm5CLFNBQW5CLEVBQXlDO0FBQ3JFLE1BQUltQixPQUFKLEVBQWE7QUFDWCxRQUFNQyxXQUFXLEdBQUdDLFNBQVMsQ0FBQ3JCLFNBQUQsQ0FBN0I7QUFDQSxXQUFPb0IsV0FBVyxDQUFDRSxTQUFaLENBQXNCLENBQXRCLEVBQXlCLEVBQXpCLElBQStCLEtBQXRDLENBRlcsQ0FFaUM7QUFDN0MsR0FIRCxNQUdPO0FBQ0wsV0FBT0QsU0FBUyxDQUFDckIsU0FBRCxDQUFoQjtBQUNEO0FBQ0YsQ0FQTSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4xMGJhY2VmNGJkOThiMDM3ODhlOC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHsgTmV4dFJvdXRlciB9IGZyb20gJ25leHQvcm91dGVyJztcclxuaW1wb3J0IFF1ZXJ5U3RyaW5nIGZyb20gJ3FzJztcclxuaW1wb3J0IHsgSW5mb1Byb3BzLCBQbG90RGF0YVByb3BzLCBRdWVyeVByb3BzIH0gZnJvbSAnLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5cclxuZXhwb3J0IGNvbnN0IHNlcGVyYXRlUnVuQW5kTHVtaUluU2VhcmNoID0gKHJ1bkFuZEx1bWk6IHN0cmluZykgPT4ge1xyXG4gIGNvbnN0IHJ1bkFuZEx1bWlBcnJheSA9IHJ1bkFuZEx1bWkuc3BsaXQoJzonKTtcclxuICBjb25zdCBwYXJzZWRSdW4gPSBydW5BbmRMdW1pQXJyYXlbMF07XHJcbiAgY29uc3QgcGFyc2VkTHVtaSA9IHJ1bkFuZEx1bWlBcnJheVsxXSA/IHBhcnNlSW50KHJ1bkFuZEx1bWlBcnJheVsxXSkgOiAwO1xyXG5cclxuICByZXR1cm4geyBwYXJzZWRSdW4sIHBhcnNlZEx1bWkgfTtcclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRfbGFiZWwgPSAoaW5mbzogSW5mb1Byb3BzLCBkYXRhPzogYW55KSA9PiB7XHJcbiAgY29uc3QgdmFsdWUgPSBkYXRhID8gZGF0YS5mU3RyaW5nIDogbnVsbDtcclxuXHJcbiAgaWYgKGluZm8/LnR5cGUgJiYgaW5mby50eXBlID09PSAndGltZScgJiYgdmFsdWUpIHtcclxuICAgIGNvbnN0IG1pbGlzZWMgPSBuZXcgRGF0ZShwYXJzZUludCh2YWx1ZSkgKiAxMDAwKTtcclxuICAgIGNvbnN0IHRpbWUgPSBtaWxpc2VjLnRvVVRDU3RyaW5nKCk7XHJcbiAgICByZXR1cm4gdGltZTtcclxuICB9IGVsc2Uge1xyXG4gICAgcmV0dXJuIHZhbHVlID8gdmFsdWUgOiAnTm8gaW5mb3JtYXRpb24nO1xyXG4gIH1cclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRQYXRoTmFtZSA9ICgpID0+IHtcclxuICBjb25zdCBpc0Jyb3dzZXIgPSAoKSA9PiB0eXBlb2Ygd2luZG93ICE9PSAndW5kZWZpbmVkJztcclxuICBsZXQgcGF0aE5hbWUgPSAoaXNCcm93c2VyKCkgJiYgd2luZG93LmxvY2F0aW9uLnBhdGhuYW1lKSB8fCAnLyc7XHJcbiAgY29uc3QgdGhlX2xhdHNfY2hhciA9IHBhdGhOYW1lLmNoYXJBdChwYXRoTmFtZS5sZW5ndGggLSAxKTtcclxuICBpZiAodGhlX2xhdHNfY2hhciAhPT0gJy8nKSB7XHJcbiAgICBwYXRoTmFtZSA9IHBhdGhOYW1lICsgJy8nXHJcbiAgfVxyXG4gIHJldHVybiBwYXRoTmFtZTtcclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBtYWtlaWQgPSAoKSA9PiB7XHJcbiAgdmFyIHRleHQgPSAnJztcclxuICB2YXIgcG9zc2libGUgPSAnQUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5eic7XHJcblxyXG4gIGZvciAodmFyIGkgPSAwOyBpIDwgNTsgaSsrKVxyXG4gICAgdGV4dCArPSBwb3NzaWJsZS5jaGFyQXQoTWF0aC5mbG9vcihNYXRoLnJhbmRvbSgpICogcG9zc2libGUubGVuZ3RoKSk7XHJcblxyXG4gIHJldHVybiB0ZXh0O1xyXG59O1xyXG5cclxuXHJcbmV4cG9ydCBjb25zdCBnZXRab29tZWRQbG90c1VybEZvck92ZXJsYXlpbmdQbG90c1dpdGhEaWZmZXJlbnROYW1lcyA9IChxdWVyeTogUXVlcnlQcm9wcywgc2VsZWN0ZWRfcGxvdDogUGxvdERhdGFQcm9wcykgPT4ge1xyXG5cclxuICBjb25zdCBwYWdlID0gJ3Bsb3RzTG9jYWxPdmVybGF5J1xyXG4gIGNvbnN0IHJ1biA9ICdydW5fbnVtYmVyPScgKyBxdWVyeS5ydW5fbnVtYmVyIGFzIHN0cmluZ1xyXG4gIGNvbnN0IGRhdGFzZXQgPSAnZGF0YXNldF9uYW1lPScgKyBxdWVyeS5kYXRhc2V0X25hbWUgYXMgc3RyaW5nXHJcbiAgY29uc3QgcGF0aCA9ICdmb2xkZXJzX3BhdGg9JyArIHNlbGVjdGVkX3Bsb3QucGF0aFxyXG4gIGNvbnN0IHBsb3RfbmFtZSA9ICdwbG90X25hbWU9JyArIHNlbGVjdGVkX3Bsb3QubmFtZVxyXG4gIGNvbnN0IHF1ZXJ5VVJMID0gW3J1biwgZGF0YXNldCwgcGF0aCwgcGxvdF9uYW1lXS5qb2luKCcmJylcclxuICBjb25zdCBwbG90c0xvY2FsT3ZlcmxheVVSTCA9IFtwYWdlLCBxdWVyeVVSTF0uam9pbignPycpXHJcbiAgcmV0dXJuIChwbG90c0xvY2FsT3ZlcmxheVVSTClcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IGdldFpvb21lZE92ZXJsYWlkUGxvdHNVcmxGb3JPdmVybGF5aW5nUGxvdHNXaXRoRGlmZmVyZW50TmFtZXMgPSAocXVlcnk6IFF1ZXJ5UHJvcHMsIHNlbGVjdGVkX3Bsb3Q6IFBsb3REYXRhUHJvcHMpID0+IHtcclxuICBjb25zdCBwYWdlID0gJ3Bsb3RzTG9jYWxPdmVybGF5J1xyXG4gIGNvbnN0IHJ1biA9ICdydW5fbnVtYmVyPScgKyBxdWVyeS5ydW5fbnVtYmVyIGFzIHN0cmluZ1xyXG4gIGNvbnN0IGRhdGFzZXQgPSAnZGF0YXNldF9uYW1lPScgKyBxdWVyeS5kYXRhc2V0X25hbWUgYXMgc3RyaW5nXHJcbiAgY29uc3QgcGF0aCA9ICdmb2xkZXJzX3BhdGg9JyArIHNlbGVjdGVkX3Bsb3QucGF0aFxyXG4gIGNvbnN0IHBsb3RfbmFtZSA9ICdwbG90X25hbWU9JyArIHNlbGVjdGVkX3Bsb3QubmFtZVxyXG4gIGNvbnN0IGdsb2JhbGx5X292ZXJsYWlkX3Bsb3RzID0gcXVlcnkub3ZlcmxheV9kYXRhPy5zcGxpdCgnJicpLm1hcCgocGxvdCkgPT4ge1xyXG4gICAgY29uc3QgcGFydHMgPSBwbG90LnNwbGl0KCcvJylcclxuICAgIGNvbnN0IHJ1bl9udW1iZXIgPSBwYXJ0cy5zaGlmdCgpXHJcbiAgICBjb25zdCBwYXRoQW5kTGFiZWwgPSBwYXJ0cy5zcGxpY2UoMylcclxuICAgIGNvbnN0IGRhdGFzZXRfbmFtZSA9IHBhcnRzLmpvaW4oJy8nKVxyXG4gICAgY29uc3QgcGF0aCA9IHNlbGVjdGVkX3Bsb3QucGF0aFxyXG4gICAgY29uc3QgcGxvdF9uYW1lID0gc2VsZWN0ZWRfcGxvdC5uYW1lXHJcbiAgICBjb25zdCBsYWJlbCA9IHBhdGhBbmRMYWJlbC5wb3AoKVxyXG4gICAgY29uc3Qgc3RyaW5nID0gW3J1bl9udW1iZXIsIGRhdGFzZXRfbmFtZSwgcGF0aCwgcGxvdF9uYW1lLCBsYWJlbF0uam9pbignLycpXHJcbiAgICByZXR1cm4gc3RyaW5nXHJcbiAgfSlcclxuICBjb25zdCBnbG9iYWxfb3ZlcmxheSA9ICdvdmVybGFpZEdsb2JhbGx5PScgKyAoZ2xvYmFsbHlfb3ZlcmxhaWRfcGxvdHMgYXMgc3RyaW5nW10pLmpvaW4oJyYnKVxyXG4gIGNvbnN0IHF1ZXJ5VVJMID0gW3J1biwgZGF0YXNldCwgcGF0aCwgcGxvdF9uYW1lLCBnbG9iYWxfb3ZlcmxheV0uam9pbignJicpXHJcbiAgY29uc3QgcGxvdHNMb2NhbE92ZXJsYXlVUkwgPSBbcGFnZSwgcXVlcnlVUkxdLmpvaW4oJz8nKVxyXG4gIHJldHVybiBwbG90c0xvY2FsT3ZlcmxheVVSTFxyXG59XHJcblxyXG5cclxuZXhwb3J0IGNvbnN0IGRlY29kZVBsb3ROYW1lID0gKHRvb0xvbmc6IGJvb2xlYW4sIHBsb3RfbmFtZTogc3RyaW5nKSA9PiB7XHJcbiAgaWYgKHRvb0xvbmcpIHtcclxuICAgIGNvbnN0IGRlY29kZV9uYW1lID0gZGVjb2RlVVJJKHBsb3RfbmFtZSlcclxuICAgIHJldHVybiBkZWNvZGVfbmFtZS5zdWJzdHJpbmcoMCwgMjUpICsgJy4uLicgLy9zb21lIG9mIG5hbWVzIGFyZSBkb3VibGUgZW5jb2RlZCBcclxuICB9IGVsc2Uge1xyXG4gICAgcmV0dXJuIGRlY29kZVVSSShwbG90X25hbWUpXHJcbiAgfVxyXG59Il0sInNvdXJjZVJvb3QiOiIifQ==