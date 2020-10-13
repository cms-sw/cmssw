webpackHotUpdate_N_E("pages/index",{

/***/ "./utils/pages/index.tsx":
/*!*******************************!*\
  !*** ./utils/pages/index.tsx ***!
  \*******************************/
/*! exports provided: navigationHandler, backToMainPage */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "navigationHandler", function() { return navigationHandler; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "backToMainPage", function() { return backToMainPage; });
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_0__);

var navigationHandler = function navigationHandler(search_by_run_number, search_by_dataset_name, search_by_lumisection) {
  next_router__WEBPACK_IMPORTED_MODULE_0___default.a.push({
    pathname: "/",
    query: {
      search_run_number: search_by_run_number,
      search_dataset_name: search_by_dataset_name,
      search_lumisection: search_by_lumisection
    }
  });
};
var backToMainPage = function backToMainPage(e) {
  next_router__WEBPACK_IMPORTED_MODULE_0___default.a.push({
    pathname: '/',
    query: {
      search_run_number: '',
      search_dataset_name: ''
    }
  });
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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vdXRpbHMvcGFnZXMvaW5kZXgudHN4Il0sIm5hbWVzIjpbIm5hdmlnYXRpb25IYW5kbGVyIiwic2VhcmNoX2J5X3J1bl9udW1iZXIiLCJzZWFyY2hfYnlfZGF0YXNldF9uYW1lIiwic2VhcmNoX2J5X2x1bWlzZWN0aW9uIiwiUm91dGVyIiwicHVzaCIsInBhdGhuYW1lIiwicXVlcnkiLCJzZWFyY2hfcnVuX251bWJlciIsInNlYXJjaF9kYXRhc2V0X25hbWUiLCJzZWFyY2hfbHVtaXNlY3Rpb24iLCJiYWNrVG9NYWluUGFnZSIsImUiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFFTyxJQUFNQSxpQkFBaUIsR0FBRyxTQUFwQkEsaUJBQW9CLENBQy9CQyxvQkFEK0IsRUFFL0JDLHNCQUYrQixFQUcvQkMscUJBSCtCLEVBSTVCO0FBQ0hDLG9EQUFNLENBQUNDLElBQVAsQ0FBWTtBQUNWQyxZQUFRLEtBREU7QUFFVkMsU0FBSyxFQUFFO0FBQ0xDLHVCQUFpQixFQUFFUCxvQkFEZDtBQUVMUSx5QkFBbUIsRUFBRVAsc0JBRmhCO0FBR0xRLHdCQUFrQixFQUFFUDtBQUhmO0FBRkcsR0FBWjtBQVFELENBYk07QUFnQkEsSUFBTVEsY0FBYyxHQUFHLFNBQWpCQSxjQUFpQixDQUFDQyxDQUFELEVBQVk7QUFDeENSLG9EQUFNLENBQUNDLElBQVAsQ0FBWTtBQUNWQyxZQUFRLEVBQUUsR0FEQTtBQUVWQyxTQUFLLEVBQUU7QUFDTEMsdUJBQWlCLEVBQUUsRUFEZDtBQUVMQyx5QkFBbUIsRUFBRTtBQUZoQjtBQUZHLEdBQVo7QUFRRCxDQVRNIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmMyZWNmOTQwOGIwYTQ4NjgxMWM5LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUm91dGVyIGZyb20gJ25leHQvcm91dGVyJztcblxuZXhwb3J0IGNvbnN0IG5hdmlnYXRpb25IYW5kbGVyID0gKFxuICBzZWFyY2hfYnlfcnVuX251bWJlcjogc3RyaW5nLFxuICBzZWFyY2hfYnlfZGF0YXNldF9uYW1lOiBzdHJpbmcsXG4gIHNlYXJjaF9ieV9sdW1pc2VjdGlvbjogc3RyaW5nXG4pID0+IHtcbiAgUm91dGVyLnB1c2goe1xuICAgIHBhdGhuYW1lOiBgL2AsXG4gICAgcXVlcnk6IHtcbiAgICAgIHNlYXJjaF9ydW5fbnVtYmVyOiBzZWFyY2hfYnlfcnVuX251bWJlcixcbiAgICAgIHNlYXJjaF9kYXRhc2V0X25hbWU6IHNlYXJjaF9ieV9kYXRhc2V0X25hbWUsXG4gICAgICBzZWFyY2hfbHVtaXNlY3Rpb246IHNlYXJjaF9ieV9sdW1pc2VjdGlvblxuICAgIH0sXG4gIH0pO1xufTtcblxuXG5leHBvcnQgY29uc3QgYmFja1RvTWFpblBhZ2UgPSAoZTogYW55KSA9PiB7XG4gIFJvdXRlci5wdXNoKHtcbiAgICBwYXRobmFtZTogJy8nLFxuICAgIHF1ZXJ5OiB7XG4gICAgICBzZWFyY2hfcnVuX251bWJlcjogJycsXG4gICAgICBzZWFyY2hfZGF0YXNldF9uYW1lOiAnJyxcbiAgICB9LFxuICB9LFxuICApXG59O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==