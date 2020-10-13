webpackHotUpdate_N_E("pages/index",{

/***/ "./containers/display/header.tsx":
/*!***************************************!*\
  !*** ./containers/display/header.tsx ***!
  \***************************************/
/*! exports provided: Header */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Header", function() { return Header; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _utils_pages__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../../utils/pages */ "./utils/pages/index.tsx");
/* harmony import */ var _components_runInfo__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../../components/runInfo */ "./components/runInfo/index.tsx");
/* harmony import */ var _components_navigation_composedSearch__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../components/navigation/composedSearch */ "./components/navigation/composedSearch.tsx");
/* harmony import */ var _components_Nav__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../components/Nav */ "./components/Nav.tsx");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/containers/display/header.tsx";

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];





var Header = function Header(_ref) {
  var isDatasetAndRunNumberSelected = _ref.isDatasetAndRunNumberSelected,
      query = _ref.query;
  return __jsx(react__WEBPACK_IMPORTED_MODULE_0__["Fragment"], null, //if all full set is selected: dataset name and run number, then regular search field is not visible.
  //Instead, run and dataset browser is is displayed.
  //Regular search fields are displayed just in the main page.
  isDatasetAndRunNumberSelected ? __jsx(react__WEBPACK_IMPORTED_MODULE_0__["Fragment"], null, __jsx(_components_runInfo__WEBPACK_IMPORTED_MODULE_2__["RunInfo"], {
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 26,
      columnNumber: 13
    }
  }), __jsx(_components_navigation_composedSearch__WEBPACK_IMPORTED_MODULE_3__["ComposedSearch"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 27,
      columnNumber: 13
    }
  })) : __jsx(react__WEBPACK_IMPORTED_MODULE_0__["Fragment"], null, __jsx(_components_Nav__WEBPACK_IMPORTED_MODULE_4__["default"], {
    initial_search_run_number: query.search_run_number,
    initial_search_dataset_name: query.search_dataset_name,
    handler: _utils_pages__WEBPACK_IMPORTED_MODULE_1__["navigationHandler"],
    type: "top",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 31,
      columnNumber: 13
    }
  })));
};
_c = Header;

var _c;

$RefreshReg$(_c, "Header");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGFpbmVycy9kaXNwbGF5L2hlYWRlci50c3giXSwibmFtZXMiOlsiSGVhZGVyIiwiaXNEYXRhc2V0QW5kUnVuTnVtYmVyU2VsZWN0ZWQiLCJxdWVyeSIsInNlYXJjaF9ydW5fbnVtYmVyIiwic2VhcmNoX2RhdGFzZXRfbmFtZSIsIm5hdmlnYXRpb25IYW5kbGVyIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQVFPLElBQU1BLE1BQU0sR0FBRyxTQUFUQSxNQUFTLE9BR0g7QUFBQSxNQUZqQkMsNkJBRWlCLFFBRmpCQSw2QkFFaUI7QUFBQSxNQURqQkMsS0FDaUIsUUFEakJBLEtBQ2lCO0FBQ2pCLFNBQ0UsNERBRUk7QUFDQTtBQUNBO0FBQ0FELCtCQUE2QixHQUMzQiw0REFDRSxNQUFDLDJEQUFEO0FBQVMsU0FBSyxFQUFFQyxLQUFoQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsRUFFRSxNQUFDLG9GQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFGRixDQUQyQixHQU0zQiw0REFDRSxNQUFDLHVEQUFEO0FBQ0UsNkJBQXlCLEVBQUVBLEtBQUssQ0FBQ0MsaUJBRG5DO0FBRUUsK0JBQTJCLEVBQUVELEtBQUssQ0FBQ0UsbUJBRnJDO0FBR0UsV0FBTyxFQUFFQyw4REFIWDtBQUlFLFFBQUksRUFBQyxLQUpQO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQVhOLENBREY7QUF3QkQsQ0E1Qk07S0FBTUwsTSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4yY2JlNmQyNzY2MTEzYWQ1OWM0OS5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xuXG5pbXBvcnQgeyBuYXZpZ2F0aW9uSGFuZGxlciB9IGZyb20gJy4uLy4uL3V0aWxzL3BhZ2VzJztcbmltcG9ydCB7IFJ1bkluZm8gfSBmcm9tICcuLi8uLi9jb21wb25lbnRzL3J1bkluZm8nO1xuaW1wb3J0IHsgQ29tcG9zZWRTZWFyY2ggfSBmcm9tICcuLi8uLi9jb21wb25lbnRzL25hdmlnYXRpb24vY29tcG9zZWRTZWFyY2gnO1xuaW1wb3J0IE5hdiBmcm9tICcuLi8uLi9jb21wb25lbnRzL05hdic7XG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi9pbnRlcmZhY2VzJztcblxuaW50ZXJmYWNlIEhlYWRlclByb3BzIHtcbiAgaXNEYXRhc2V0QW5kUnVuTnVtYmVyU2VsZWN0ZWQ6IGJvb2xlYW47XG4gIHF1ZXJ5OiBRdWVyeVByb3BzO1xufVxuXG5leHBvcnQgY29uc3QgSGVhZGVyID0gKHtcbiAgaXNEYXRhc2V0QW5kUnVuTnVtYmVyU2VsZWN0ZWQsXG4gIHF1ZXJ5LFxufTogSGVhZGVyUHJvcHMpID0+IHtcbiAgcmV0dXJuIChcbiAgICA8PlxuICAgICAge1xuICAgICAgICAvL2lmIGFsbCBmdWxsIHNldCBpcyBzZWxlY3RlZDogZGF0YXNldCBuYW1lIGFuZCBydW4gbnVtYmVyLCB0aGVuIHJlZ3VsYXIgc2VhcmNoIGZpZWxkIGlzIG5vdCB2aXNpYmxlLlxuICAgICAgICAvL0luc3RlYWQsIHJ1biBhbmQgZGF0YXNldCBicm93c2VyIGlzIGlzIGRpc3BsYXllZC5cbiAgICAgICAgLy9SZWd1bGFyIHNlYXJjaCBmaWVsZHMgYXJlIGRpc3BsYXllZCBqdXN0IGluIHRoZSBtYWluIHBhZ2UuXG4gICAgICAgIGlzRGF0YXNldEFuZFJ1bk51bWJlclNlbGVjdGVkID8gKFxuICAgICAgICAgIDw+XG4gICAgICAgICAgICA8UnVuSW5mbyBxdWVyeT17cXVlcnl9IC8+XG4gICAgICAgICAgICA8Q29tcG9zZWRTZWFyY2ggLz5cbiAgICAgICAgICA8Lz5cbiAgICAgICAgKSA6IChcbiAgICAgICAgICA8PlxuICAgICAgICAgICAgPE5hdlxuICAgICAgICAgICAgICBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyPXtxdWVyeS5zZWFyY2hfcnVuX251bWJlcn1cbiAgICAgICAgICAgICAgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lPXtxdWVyeS5zZWFyY2hfZGF0YXNldF9uYW1lfVxuICAgICAgICAgICAgICBoYW5kbGVyPXtuYXZpZ2F0aW9uSGFuZGxlcn1cbiAgICAgICAgICAgICAgdHlwZT1cInRvcFwiXG4gICAgICAgICAgICAvPlxuICAgICAgICAgIDwvPlxuICAgICAgICApXG4gICAgICB9XG4gICAgPC8+XG4gICk7XG59O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==