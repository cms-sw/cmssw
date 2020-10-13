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
/* harmony import */ var clean_deep__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! clean-deep */ "./node_modules/clean-deep/src/index.js");
/* harmony import */ var clean_deep__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(clean_deep__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_1__);


var navigationHandler = function navigationHandler(search_by_run_number, search_by_dataset_name, search_by_lumisection) {
  var params = clean_deep__WEBPACK_IMPORTED_MODULE_0___default()({
    search_run_number: search_by_run_number,
    search_dataset_name: search_by_dataset_name,
    search_lumisection: search_by_lumisection
  });
  next_router__WEBPACK_IMPORTED_MODULE_1___default.a.push({
    pathname: "/",
    query: params
  });
};
var backToMainPage = function backToMainPage(e) {
  next_router__WEBPACK_IMPORTED_MODULE_1___default.a.push({
    pathname: '/' // query: {
    //   search_run_number: '',
    //   search_dataset_name: '',
    // },

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vdXRpbHMvcGFnZXMvaW5kZXgudHN4Il0sIm5hbWVzIjpbIm5hdmlnYXRpb25IYW5kbGVyIiwic2VhcmNoX2J5X3J1bl9udW1iZXIiLCJzZWFyY2hfYnlfZGF0YXNldF9uYW1lIiwic2VhcmNoX2J5X2x1bWlzZWN0aW9uIiwicGFyYW1zIiwiY2xlYW5EZWVwIiwic2VhcmNoX3J1bl9udW1iZXIiLCJzZWFyY2hfZGF0YXNldF9uYW1lIiwic2VhcmNoX2x1bWlzZWN0aW9uIiwiUm91dGVyIiwicHVzaCIsInBhdGhuYW1lIiwicXVlcnkiLCJiYWNrVG9NYWluUGFnZSIsImUiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQ0E7QUFFTyxJQUFNQSxpQkFBaUIsR0FBRyxTQUFwQkEsaUJBQW9CLENBQy9CQyxvQkFEK0IsRUFFL0JDLHNCQUYrQixFQUcvQkMscUJBSCtCLEVBSTVCO0FBQ0gsTUFBTUMsTUFBTSxHQUFHQyxpREFBUyxDQUFDO0FBQ3ZCQyxxQkFBaUIsRUFBRUwsb0JBREk7QUFFdkJNLHVCQUFtQixFQUFFTCxzQkFGRTtBQUd2Qk0sc0JBQWtCLEVBQUVMO0FBSEcsR0FBRCxDQUF4QjtBQUtBTSxvREFBTSxDQUFDQyxJQUFQLENBQVk7QUFDVkMsWUFBUSxLQURFO0FBRVZDLFNBQUssRUFBRVI7QUFGRyxHQUFaO0FBSUQsQ0FkTTtBQWlCQSxJQUFNUyxjQUFjLEdBQUcsU0FBakJBLGNBQWlCLENBQUNDLENBQUQsRUFBWTtBQUN4Q0wsb0RBQU0sQ0FBQ0MsSUFBUCxDQUFZO0FBQ1ZDLFlBQVEsRUFBRSxHQURBLENBRVY7QUFDQTtBQUNBO0FBQ0E7O0FBTFUsR0FBWjtBQVFELENBVE0iLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNzU3OWVhZjNmMzQ4Y2Y4NDk1ODQuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBjbGVhbkRlZXAgZnJvbSAnY2xlYW4tZGVlcCc7XG5pbXBvcnQgUm91dGVyIGZyb20gJ25leHQvcm91dGVyJztcblxuZXhwb3J0IGNvbnN0IG5hdmlnYXRpb25IYW5kbGVyID0gKFxuICBzZWFyY2hfYnlfcnVuX251bWJlcjogc3RyaW5nLFxuICBzZWFyY2hfYnlfZGF0YXNldF9uYW1lOiBzdHJpbmcsXG4gIHNlYXJjaF9ieV9sdW1pc2VjdGlvbjogc3RyaW5nXG4pID0+IHtcbiAgY29uc3QgcGFyYW1zID0gY2xlYW5EZWVwKHtcbiAgICBzZWFyY2hfcnVuX251bWJlcjogc2VhcmNoX2J5X3J1bl9udW1iZXIsXG4gICAgc2VhcmNoX2RhdGFzZXRfbmFtZTogc2VhcmNoX2J5X2RhdGFzZXRfbmFtZSxcbiAgICBzZWFyY2hfbHVtaXNlY3Rpb246IHNlYXJjaF9ieV9sdW1pc2VjdGlvblxuICB9KVxuICBSb3V0ZXIucHVzaCh7XG4gICAgcGF0aG5hbWU6IGAvYCxcbiAgICBxdWVyeTogcGFyYW1zLFxuICB9KTtcbn07XG5cblxuZXhwb3J0IGNvbnN0IGJhY2tUb01haW5QYWdlID0gKGU6IGFueSkgPT4ge1xuICBSb3V0ZXIucHVzaCh7XG4gICAgcGF0aG5hbWU6ICcvJyxcbiAgICAvLyBxdWVyeToge1xuICAgIC8vICAgc2VhcmNoX3J1bl9udW1iZXI6ICcnLFxuICAgIC8vICAgc2VhcmNoX2RhdGFzZXRfbmFtZTogJycsXG4gICAgLy8gfSxcbiAgfSxcbiAgKVxufTtcbiJdLCJzb3VyY2VSb290IjoiIn0=