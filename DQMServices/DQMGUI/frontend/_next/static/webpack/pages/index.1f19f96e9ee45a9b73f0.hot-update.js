webpackHotUpdate_N_E("pages/index",{

/***/ "./components/workspaces/utils.ts":
/*!****************************************!*\
  !*** ./components/workspaces/utils.ts ***!
  \****************************************/
/*! exports provided: setWorkspaceToQuery, removeFirstSlash */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "setWorkspaceToQuery", function() { return setWorkspaceToQuery; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "removeFirstSlash", function() { return removeFirstSlash; });
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_0__);

var setWorkspaceToQuery = function setWorkspaceToQuery(query, workspace) {
  return next_router__WEBPACK_IMPORTED_MODULE_0___default.a.push({
    pathname: '/',
    query: {
      run_number: query.run_number,
      dataset_name: query.dataset_name,
      workspaces: workspace
    }
  });
};
var removeFirstSlash = function removeFirstSlash(path) {
  var firstChar = path.substring(0, 1);

  if (firstChar === '/') {
    return path.substring(1, path.length);
  } else {
    return path;
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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy93b3Jrc3BhY2VzL3V0aWxzLnRzIl0sIm5hbWVzIjpbInNldFdvcmtzcGFjZVRvUXVlcnkiLCJxdWVyeSIsIndvcmtzcGFjZSIsIlJvdXRlciIsInB1c2giLCJwYXRobmFtZSIsInJ1bl9udW1iZXIiLCJkYXRhc2V0X25hbWUiLCJ3b3Jrc3BhY2VzIiwicmVtb3ZlRmlyc3RTbGFzaCIsInBhdGgiLCJmaXJzdENoYXIiLCJzdWJzdHJpbmciLCJsZW5ndGgiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7QUFHQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFFTyxJQUFNQSxtQkFBbUIsR0FBRyxTQUF0QkEsbUJBQXNCLENBQUNDLEtBQUQsRUFBb0JDLFNBQXBCLEVBQTBDO0FBQzNFLFNBQU9DLGtEQUFNLENBQUNDLElBQVAsQ0FBWTtBQUNqQkMsWUFBUSxFQUFFLEdBRE87QUFFakJKLFNBQUssRUFBRTtBQUNMSyxnQkFBVSxFQUFFTCxLQUFLLENBQUNLLFVBRGI7QUFFTEMsa0JBQVksRUFBRU4sS0FBSyxDQUFDTSxZQUZmO0FBR0xDLGdCQUFVLEVBQUVOO0FBSFA7QUFGVSxHQUFaLENBQVA7QUFRRCxDQVRNO0FBV0EsSUFBTU8sZ0JBQWdCLEdBQUcsU0FBbkJBLGdCQUFtQixDQUFDQyxJQUFELEVBQWtCO0FBQ2hELE1BQU1DLFNBQVMsR0FBR0QsSUFBSSxDQUFDRSxTQUFMLENBQWUsQ0FBZixFQUFrQixDQUFsQixDQUFsQjs7QUFDQSxNQUFJRCxTQUFTLEtBQUssR0FBbEIsRUFBdUI7QUFDckIsV0FBT0QsSUFBSSxDQUFDRSxTQUFMLENBQWUsQ0FBZixFQUFrQkYsSUFBSSxDQUFDRyxNQUF2QixDQUFQO0FBQ0QsR0FGRCxNQUVPO0FBQ0wsV0FBT0gsSUFBUDtBQUNEO0FBQ0YsQ0FQTSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4xZjE5Zjk2ZTllZTQ1YTliNzNmMC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHFzIGZyb20gJ3FzJztcblxuaW1wb3J0IHsgUXVlcnlQcm9wcyB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcbmltcG9ydCBSb3V0ZXIgZnJvbSAnbmV4dC9yb3V0ZXInO1xuXG5leHBvcnQgY29uc3Qgc2V0V29ya3NwYWNlVG9RdWVyeSA9IChxdWVyeTogUXVlcnlQcm9wcywgd29ya3NwYWNlOiBzdHJpbmcpID0+IHtcbiAgcmV0dXJuIFJvdXRlci5wdXNoKHtcbiAgICBwYXRobmFtZTogJy8nLFxuICAgIHF1ZXJ5OiB7XG4gICAgICBydW5fbnVtYmVyOiBxdWVyeS5ydW5fbnVtYmVyLFxuICAgICAgZGF0YXNldF9uYW1lOiBxdWVyeS5kYXRhc2V0X25hbWUsXG4gICAgICB3b3Jrc3BhY2VzOiB3b3Jrc3BhY2UsXG4gICAgfSxcbiAgfSk7XG59O1xuXG5leHBvcnQgY29uc3QgcmVtb3ZlRmlyc3RTbGFzaCA9IChwYXRoOiBzdHJpbmcpID0+IHtcbiAgY29uc3QgZmlyc3RDaGFyID0gcGF0aC5zdWJzdHJpbmcoMCwgMSk7XG4gIGlmIChmaXJzdENoYXIgPT09ICcvJykge1xuICAgIHJldHVybiBwYXRoLnN1YnN0cmluZygxLCBwYXRoLmxlbmd0aCk7XG4gIH0gZWxzZSB7XG4gICAgcmV0dXJuIHBhdGg7XG4gIH1cbn07XG4iXSwic291cmNlUm9vdCI6IiJ9