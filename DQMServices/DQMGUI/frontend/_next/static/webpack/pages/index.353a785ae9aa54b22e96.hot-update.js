webpackHotUpdate_N_E("pages/index",{

/***/ "./components/utils.ts":
/*!*****************************!*\
  !*** ./components/utils.ts ***!
  \*****************************/
/*! exports provided: seperateRunAndLumiInSearch, get_label, getPathName, makeid */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "seperateRunAndLumiInSearch", function() { return seperateRunAndLumiInSearch; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_label", function() { return get_label; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getPathName", function() { return getPathName; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "makeid", function() { return makeid; });
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

  if ((info === null || info === void 0 ? void 0 : info.type) && info.type === 'time' && value) {
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
  var removedTheLastSlash = pathName.substring(0, pathName.length - 1);
  console.log(removedTheLastSlash);
  return removedTheLastSlash;
};
var makeid = function makeid() {
  var text = '';
  var possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';

  for (var i = 0; i < 5; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }

  return text;
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy91dGlscy50cyJdLCJuYW1lcyI6WyJzZXBlcmF0ZVJ1bkFuZEx1bWlJblNlYXJjaCIsInJ1bkFuZEx1bWkiLCJydW5BbmRMdW1pQXJyYXkiLCJzcGxpdCIsInBhcnNlZFJ1biIsInBhcnNlZEx1bWkiLCJwYXJzZUludCIsImdldF9sYWJlbCIsImluZm8iLCJkYXRhIiwidmFsdWUiLCJmU3RyaW5nIiwidHlwZSIsIm1pbGlzZWMiLCJEYXRlIiwidGltZSIsInRvVVRDU3RyaW5nIiwiZ2V0UGF0aE5hbWUiLCJpc0Jyb3dzZXIiLCJwYXRoTmFtZSIsIndpbmRvdyIsImxvY2F0aW9uIiwicGF0aG5hbWUiLCJyZW1vdmVkVGhlTGFzdFNsYXNoIiwic3Vic3RyaW5nIiwibGVuZ3RoIiwiY29uc29sZSIsImxvZyIsIm1ha2VpZCIsInRleHQiLCJwb3NzaWJsZSIsImkiLCJjaGFyQXQiLCJNYXRoIiwiZmxvb3IiLCJyYW5kb20iXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7QUFFQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQU8sSUFBTUEsMEJBQTBCLEdBQUcsU0FBN0JBLDBCQUE2QixDQUFDQyxVQUFELEVBQXdCO0FBQ2hFLE1BQU1DLGVBQWUsR0FBR0QsVUFBVSxDQUFDRSxLQUFYLENBQWlCLEdBQWpCLENBQXhCO0FBQ0EsTUFBTUMsU0FBUyxHQUFHRixlQUFlLENBQUMsQ0FBRCxDQUFqQztBQUNBLE1BQU1HLFVBQVUsR0FBR0gsZUFBZSxDQUFDLENBQUQsQ0FBZixHQUFxQkksUUFBUSxDQUFDSixlQUFlLENBQUMsQ0FBRCxDQUFoQixDQUE3QixHQUFvRCxDQUF2RTtBQUVBLFNBQU87QUFBRUUsYUFBUyxFQUFUQSxTQUFGO0FBQWFDLGNBQVUsRUFBVkE7QUFBYixHQUFQO0FBQ0QsQ0FOTTtBQVFBLElBQU1FLFNBQVMsR0FBRyxTQUFaQSxTQUFZLENBQUNDLElBQUQsRUFBa0JDLElBQWxCLEVBQWlDO0FBQ3hELE1BQU1DLEtBQUssR0FBR0QsSUFBSSxHQUFHQSxJQUFJLENBQUNFLE9BQVIsR0FBa0IsSUFBcEM7O0FBRUEsTUFBSSxDQUFBSCxJQUFJLFNBQUosSUFBQUEsSUFBSSxXQUFKLFlBQUFBLElBQUksQ0FBRUksSUFBTixLQUFjSixJQUFJLENBQUNJLElBQUwsS0FBYyxNQUE1QixJQUFzQ0YsS0FBMUMsRUFBaUQ7QUFDL0MsUUFBTUcsT0FBTyxHQUFHLElBQUlDLElBQUosQ0FBU1IsUUFBUSxDQUFDSSxLQUFELENBQVIsR0FBa0IsSUFBM0IsQ0FBaEI7QUFDQSxRQUFNSyxJQUFJLEdBQUdGLE9BQU8sQ0FBQ0csV0FBUixFQUFiO0FBQ0EsV0FBT0QsSUFBUDtBQUNELEdBSkQsTUFJTztBQUNMLFdBQU9MLEtBQUssR0FBR0EsS0FBSCxHQUFXLGdCQUF2QjtBQUNEO0FBQ0YsQ0FWTTtBQVlBLElBQU1PLFdBQVcsR0FBRyxTQUFkQSxXQUFjLEdBQU07QUFDL0IsTUFBTUMsU0FBUyxHQUFHLFNBQVpBLFNBQVk7QUFBQTtBQUFBLEdBQWxCOztBQUNBLE1BQU1DLFFBQVEsR0FBSUQsU0FBUyxNQUFNRSxNQUFNLENBQUNDLFFBQVAsQ0FBZ0JDLFFBQWhDLElBQTZDLEdBQTlEO0FBQ0EsTUFBTUMsbUJBQW1CLEdBQUdKLFFBQVEsQ0FBQ0ssU0FBVCxDQUFtQixDQUFuQixFQUFzQkwsUUFBUSxDQUFDTSxNQUFULEdBQWtCLENBQXhDLENBQTVCO0FBQ0FDLFNBQU8sQ0FBQ0MsR0FBUixDQUFZSixtQkFBWjtBQUNBLFNBQU9BLG1CQUFQO0FBQ0QsQ0FOTTtBQU9BLElBQU1LLE1BQU0sR0FBRyxTQUFUQSxNQUFTLEdBQU07QUFDMUIsTUFBSUMsSUFBSSxHQUFHLEVBQVg7QUFDQSxNQUFJQyxRQUFRLEdBQUcsc0RBQWY7O0FBRUEsT0FBSyxJQUFJQyxDQUFDLEdBQUcsQ0FBYixFQUFnQkEsQ0FBQyxHQUFHLENBQXBCLEVBQXVCQSxDQUFDLEVBQXhCO0FBQ0VGLFFBQUksSUFBSUMsUUFBUSxDQUFDRSxNQUFULENBQWdCQyxJQUFJLENBQUNDLEtBQUwsQ0FBV0QsSUFBSSxDQUFDRSxNQUFMLEtBQWdCTCxRQUFRLENBQUNMLE1BQXBDLENBQWhCLENBQVI7QUFERjs7QUFHQSxTQUFPSSxJQUFQO0FBQ0QsQ0FSTSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4zNTNhNzg1YWU5YWE1NGIyMmU5Ni5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHsgSW5mb1Byb3BzIH0gZnJvbSAnLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuXG5leHBvcnQgY29uc3Qgc2VwZXJhdGVSdW5BbmRMdW1pSW5TZWFyY2ggPSAocnVuQW5kTHVtaTogc3RyaW5nKSA9PiB7XG4gIGNvbnN0IHJ1bkFuZEx1bWlBcnJheSA9IHJ1bkFuZEx1bWkuc3BsaXQoJzonKTtcbiAgY29uc3QgcGFyc2VkUnVuID0gcnVuQW5kTHVtaUFycmF5WzBdO1xuICBjb25zdCBwYXJzZWRMdW1pID0gcnVuQW5kTHVtaUFycmF5WzFdID8gcGFyc2VJbnQocnVuQW5kTHVtaUFycmF5WzFdKSA6IDA7XG5cbiAgcmV0dXJuIHsgcGFyc2VkUnVuLCBwYXJzZWRMdW1pIH07XG59O1xuXG5leHBvcnQgY29uc3QgZ2V0X2xhYmVsID0gKGluZm86IEluZm9Qcm9wcywgZGF0YT86IGFueSkgPT4ge1xuICBjb25zdCB2YWx1ZSA9IGRhdGEgPyBkYXRhLmZTdHJpbmcgOiBudWxsO1xuXG4gIGlmIChpbmZvPy50eXBlICYmIGluZm8udHlwZSA9PT0gJ3RpbWUnICYmIHZhbHVlKSB7XG4gICAgY29uc3QgbWlsaXNlYyA9IG5ldyBEYXRlKHBhcnNlSW50KHZhbHVlKSAqIDEwMDApO1xuICAgIGNvbnN0IHRpbWUgPSBtaWxpc2VjLnRvVVRDU3RyaW5nKCk7XG4gICAgcmV0dXJuIHRpbWU7XG4gIH0gZWxzZSB7XG4gICAgcmV0dXJuIHZhbHVlID8gdmFsdWUgOiAnTm8gaW5mb3JtYXRpb24nO1xuICB9XG59O1xuXG5leHBvcnQgY29uc3QgZ2V0UGF0aE5hbWUgPSAoKSA9PiB7XG4gIGNvbnN0IGlzQnJvd3NlciA9ICgpID0+IHR5cGVvZiB3aW5kb3cgIT09ICd1bmRlZmluZWQnO1xuICBjb25zdCBwYXRoTmFtZSA9IChpc0Jyb3dzZXIoKSAmJiB3aW5kb3cubG9jYXRpb24ucGF0aG5hbWUpIHx8ICcvJztcbiAgY29uc3QgcmVtb3ZlZFRoZUxhc3RTbGFzaCA9IHBhdGhOYW1lLnN1YnN0cmluZygwLCBwYXRoTmFtZS5sZW5ndGggLSAxKTtcbiAgY29uc29sZS5sb2cocmVtb3ZlZFRoZUxhc3RTbGFzaClcbiAgcmV0dXJuIHJlbW92ZWRUaGVMYXN0U2xhc2g7XG59O1xuZXhwb3J0IGNvbnN0IG1ha2VpZCA9ICgpID0+IHtcbiAgdmFyIHRleHQgPSAnJztcbiAgdmFyIHBvc3NpYmxlID0gJ0FCQ0RFRkdISUpLTE1OT1BRUlNUVVZXWFlaYWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXonO1xuXG4gIGZvciAodmFyIGkgPSAwOyBpIDwgNTsgaSsrKVxuICAgIHRleHQgKz0gcG9zc2libGUuY2hhckF0KE1hdGguZmxvb3IoTWF0aC5yYW5kb20oKSAqIHBvc3NpYmxlLmxlbmd0aCkpO1xuXG4gIHJldHVybiB0ZXh0O1xufTtcbiJdLCJzb3VyY2VSb290IjoiIn0=