webpackHotUpdate_N_E("pages/index",{

/***/ "./hooks/useChangeRouter.tsx":
/*!***********************************!*\
  !*** ./hooks/useChangeRouter.tsx ***!
  \***********************************/
/*! exports provided: useChangeRouter */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "useChangeRouter", function() { return useChangeRouter; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var qs__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! qs */ "./node_modules/qs/lib/index.js");
/* harmony import */ var qs__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(qs__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../containers/display/utils */ "./containers/display/utils.ts");
var _s = $RefreshSig$();





var useChangeRouter = function useChangeRouter(params) {
  _s();

  var watchers = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : [];
  var condition = arguments.length > 2 ? arguments[2] : undefined;
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_1__["useRouter"])();
  var query = router.query;
  var parameters = Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_3__["getChangedQueryParams"])(params, query);
  var queryString = qs__WEBPACK_IMPORTED_MODULE_2___default.a.stringify(parameters, {});
  react__WEBPACK_IMPORTED_MODULE_0__["useEffect"](function () {
    if (condition) {
      next_router__WEBPACK_IMPORTED_MODULE_1___default.a.push({
        pathname: "/",
        query: parameters,
        path: decodeURIComponent(queryString)
      });
    }
  }, watchers);
};

_s(useChangeRouter, "vQduR7x+OPXj6PSmJyFnf+hU7bg=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_1__["useRouter"]];
});

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vaG9va3MvdXNlQ2hhbmdlUm91dGVyLnRzeCJdLCJuYW1lcyI6WyJ1c2VDaGFuZ2VSb3V0ZXIiLCJwYXJhbXMiLCJ3YXRjaGVycyIsImNvbmRpdGlvbiIsInJvdXRlciIsInVzZVJvdXRlciIsInF1ZXJ5IiwicGFyYW1ldGVycyIsImdldENoYW5nZWRRdWVyeVBhcmFtcyIsInF1ZXJ5U3RyaW5nIiwicXMiLCJzdHJpbmdpZnkiLCJSZWFjdCIsIlJvdXRlciIsInB1c2giLCJwYXRobmFtZSIsInBhdGgiLCJkZWNvZGVVUklDb21wb25lbnQiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUlBO0FBRU8sSUFBTUEsZUFBZSxHQUFHLFNBQWxCQSxlQUFrQixDQUM3QkMsTUFENkIsRUFJMUI7QUFBQTs7QUFBQSxNQUZIQyxRQUVHLHVFQUZ5QyxFQUV6QztBQUFBLE1BREhDLFNBQ0c7QUFDSCxNQUFNQyxNQUFNLEdBQUdDLDZEQUFTLEVBQXhCO0FBQ0EsTUFBTUMsS0FBaUIsR0FBR0YsTUFBTSxDQUFDRSxLQUFqQztBQUVBLE1BQU1DLFVBQVUsR0FBR0MsdUZBQXFCLENBQUNQLE1BQUQsRUFBU0ssS0FBVCxDQUF4QztBQUNBLE1BQU1HLFdBQVcsR0FBR0MseUNBQUUsQ0FBQ0MsU0FBSCxDQUFhSixVQUFiLEVBQXlCLEVBQXpCLENBQXBCO0FBRUFLLGlEQUFBLENBQWdCLFlBQU07QUFDcEIsUUFBSVQsU0FBSixFQUFlO0FBQ2JVLHdEQUFNLENBQUNDLElBQVAsQ0FBWTtBQUNWQyxnQkFBUSxLQURFO0FBRVZULGFBQUssRUFBRUMsVUFGRztBQUdWUyxZQUFJLEVBQUVDLGtCQUFrQixDQUFDUixXQUFEO0FBSGQsT0FBWjtBQUtEO0FBQ0YsR0FSRCxFQVFHUCxRQVJIO0FBU0QsQ0FwQk07O0dBQU1GLGU7VUFLSUsscUQiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguN2NkNGIyOWE2MzI1MjQ1YzhjODYuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCBSb3V0ZXIsIHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xuaW1wb3J0IHFzIGZyb20gJ3FzJztcblxuaW1wb3J0IHsgUXVlcnlQcm9wcyB9IGZyb20gJy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcbmltcG9ydCB7IFBhcnNlZFVybFF1ZXJ5SW5wdXQgfSBmcm9tICdxdWVyeXN0cmluZyc7XG5pbXBvcnQgeyBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMgfSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvdXRpbHMnO1xuXG5leHBvcnQgY29uc3QgdXNlQ2hhbmdlUm91dGVyID0gKFxuICBwYXJhbXM6IFBhcnNlZFVybFF1ZXJ5SW5wdXQsXG4gIHdhdGNoZXJzOiAoc3RyaW5nIHwgbnVtYmVyIHwgdW5kZWZpbmVkKVtdID0gW10sXG4gIGNvbmRpdGlvbjogYm9vbGVhblxuKSA9PiB7XG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcblxuICBjb25zdCBwYXJhbWV0ZXJzID0gZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zKHBhcmFtcywgcXVlcnkpO1xuICBjb25zdCBxdWVyeVN0cmluZyA9IHFzLnN0cmluZ2lmeShwYXJhbWV0ZXJzLCB7fSk7XG5cbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcbiAgICBpZiAoY29uZGl0aW9uKSB7XG4gICAgICBSb3V0ZXIucHVzaCh7XG4gICAgICAgIHBhdGhuYW1lOiBgL2AsXG4gICAgICAgIHF1ZXJ5OiBwYXJhbWV0ZXJzLFxuICAgICAgICBwYXRoOiBkZWNvZGVVUklDb21wb25lbnQocXVlcnlTdHJpbmcpLFxuICAgICAgfSk7XG4gICAgfVxuICB9LCB3YXRjaGVycyk7XG59O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==