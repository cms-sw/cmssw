webpackHotUpdate_N_E("pages/index",{

/***/ "./components/modes/modesSelection.tsx":
/*!*********************************************!*\
  !*** ./components/modes/modesSelection.tsx ***!
  \*********************************************/
/*! exports provided: ModesSelection */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ModesSelection", function() { return ModesSelection; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/modes/modesSelection.tsx",
    _this = undefined;

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];




var modes = [{
  name: 'Online',
  link: 'https://cmsweb.cern.ch/dqm/online-new/'
}, {
  name: 'Online-playback',
  link: 'https://cmsweb.cern.ch/dqm/online-playback-new/'
}, {
  name: 'offline',
  link: 'https://dqm-gui.web.cern.ch/'
}];

var menu = __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"], {
  __self: undefined,
  __source: {
    fileName: _jsxFileName,
    lineNumber: 20,
    columnNumber: 3
  }
}, modes.map(function (mode) {
  return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"].Item, {
    disabled: _config_config__WEBPACK_IMPORTED_MODULE_2__["functions_config"].mode.toUpperCase() === mode.name.toUpperCase(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 22,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    type: "link",
    disabled: _config_config__WEBPACK_IMPORTED_MODULE_2__["functions_config"].mode.toUpperCase() === mode.name.toUpperCase(),
    onClick: function onClick() {
      return location.href = mode.link;
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 24,
      columnNumber: 9
    }
  }, mode.name));
}));

var ModesSelection = function ModesSelection() {
  return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Dropdown"], {
    overlay: menu,
    placement: "bottomCenter",
    arrow: true,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 38,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    type: "link",
    style: {
      color: 'white',
      fontVariant: 'all-small-caps'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 39,
      columnNumber: 7
    }
  }, _config_config__WEBPACK_IMPORTED_MODULE_2__["mode"]));
};
_c = ModesSelection;

var _c;

$RefreshReg$(_c, "ModesSelection");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9tb2Rlcy9tb2Rlc1NlbGVjdGlvbi50c3giXSwibmFtZXMiOlsibW9kZXMiLCJuYW1lIiwibGluayIsIm1lbnUiLCJtYXAiLCJtb2RlIiwiZnVuY3Rpb25zX2NvbmZpZyIsInRvVXBwZXJDYXNlIiwibG9jYXRpb24iLCJocmVmIiwiTW9kZXNTZWxlY3Rpb24iLCJjb2xvciIsImZvbnRWYXJpYW50Il0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUNBO0FBRUEsSUFBTUEsS0FBSyxHQUFHLENBQUM7QUFDYkMsTUFBSSxFQUFFLFFBRE87QUFFYkMsTUFBSSxFQUFFO0FBRk8sQ0FBRCxFQUlkO0FBQ0VELE1BQUksRUFBRSxpQkFEUjtBQUVFQyxNQUFJLEVBQUU7QUFGUixDQUpjLEVBUWQ7QUFDRUQsTUFBSSxFQUFFLFNBRFI7QUFFRUMsTUFBSSxFQUFFO0FBRlIsQ0FSYyxDQUFkOztBQWFBLElBQU1DLElBQUksR0FDUixNQUFDLHlDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsR0FDR0gsS0FBSyxDQUFDSSxHQUFOLENBQVUsVUFBQUMsSUFBSTtBQUFBLFNBQ2IsTUFBQyx5Q0FBRCxDQUFNLElBQU47QUFDRSxZQUFRLEVBQUVDLCtEQUFnQixDQUFDRCxJQUFqQixDQUFzQkUsV0FBdEIsT0FBd0NGLElBQUksQ0FBQ0osSUFBTCxDQUFVTSxXQUFWLEVBRHBEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FFRSxNQUFDLDJDQUFEO0FBQ0UsUUFBSSxFQUFDLE1BRFA7QUFFRSxZQUFRLEVBQUVELCtEQUFnQixDQUFDRCxJQUFqQixDQUFzQkUsV0FBdEIsT0FBd0NGLElBQUksQ0FBQ0osSUFBTCxDQUFVTSxXQUFWLEVBRnBEO0FBR0UsV0FBTyxFQUFFO0FBQUEsYUFBTUMsUUFBUSxDQUFDQyxJQUFULEdBQWdCSixJQUFJLENBQUNILElBQTNCO0FBQUEsS0FIWDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBSUdHLElBQUksQ0FBQ0osSUFKUixDQUZGLENBRGE7QUFBQSxDQUFkLENBREgsQ0FERjs7QUFpQk8sSUFBTVMsY0FBYyxHQUFHLFNBQWpCQSxjQUFpQixHQUFNO0FBQ2xDLFNBQ0UsTUFBQyw2Q0FBRDtBQUFVLFdBQU8sRUFBRVAsSUFBbkI7QUFBeUIsYUFBUyxFQUFDLGNBQW5DO0FBQWtELFNBQUssTUFBdkQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsMkNBQUQ7QUFDRSxRQUFJLEVBQUMsTUFEUDtBQUVFLFNBQUssRUFBRTtBQUFFUSxXQUFLLEVBQUUsT0FBVDtBQUFrQkMsaUJBQVcsRUFBRTtBQUEvQixLQUZUO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FHR1AsbURBSEgsQ0FERixDQURGO0FBVUQsQ0FYTTtLQUFNSyxjIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjE2NTQxMDgyYzhiZjNmZTljYzNjLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IEJ1dHRvbiB9IGZyb20gJ2FudGQnO1xyXG5pbXBvcnQgeyBNZW51LCBEcm9wZG93biB9IGZyb20gJ2FudGQnO1xyXG5pbXBvcnQgeyBmdW5jdGlvbnNfY29uZmlnLCBtb2RlIH0gZnJvbSAnLi4vLi4vY29uZmlnL2NvbmZpZyc7XHJcblxyXG5jb25zdCBtb2RlcyA9IFt7XHJcbiAgbmFtZTogJ09ubGluZScsXHJcbiAgbGluazogJ2h0dHBzOi8vY21zd2ViLmNlcm4uY2gvZHFtL29ubGluZS1uZXcvJ1xyXG59LFxyXG57XHJcbiAgbmFtZTogJ09ubGluZS1wbGF5YmFjaycsXHJcbiAgbGluazogJ2h0dHBzOi8vY21zd2ViLmNlcm4uY2gvZHFtL29ubGluZS1wbGF5YmFjay1uZXcvJ1xyXG59LFxyXG57XHJcbiAgbmFtZTogJ29mZmxpbmUnLFxyXG4gIGxpbms6ICdodHRwczovL2RxbS1ndWkud2ViLmNlcm4uY2gvJ1xyXG59XVxyXG5cclxuY29uc3QgbWVudSA9IChcclxuICA8TWVudT5cclxuICAgIHttb2Rlcy5tYXAobW9kZSA9PiAoXHJcbiAgICAgIDxNZW51Lkl0ZW1cclxuICAgICAgICBkaXNhYmxlZD17ZnVuY3Rpb25zX2NvbmZpZy5tb2RlLnRvVXBwZXJDYXNlKCkgPT09IG1vZGUubmFtZS50b1VwcGVyQ2FzZSgpfT5cclxuICAgICAgICA8QnV0dG9uXHJcbiAgICAgICAgICB0eXBlPVwibGlua1wiXHJcbiAgICAgICAgICBkaXNhYmxlZD17ZnVuY3Rpb25zX2NvbmZpZy5tb2RlLnRvVXBwZXJDYXNlKCkgPT09IG1vZGUubmFtZS50b1VwcGVyQ2FzZSgpfVxyXG4gICAgICAgICAgb25DbGljaz17KCkgPT4gbG9jYXRpb24uaHJlZiA9IG1vZGUubGlua30gPlxyXG4gICAgICAgICAge21vZGUubmFtZX1cclxuICAgICAgICA8L0J1dHRvbj5cclxuICAgICAgPC9NZW51Lkl0ZW0+XHJcbiAgICApKX1cclxuICA8L01lbnU+XHJcbik7XHJcblxyXG5cclxuZXhwb3J0IGNvbnN0IE1vZGVzU2VsZWN0aW9uID0gKCkgPT4ge1xyXG4gIHJldHVybiAoXHJcbiAgICA8RHJvcGRvd24gb3ZlcmxheT17bWVudX0gcGxhY2VtZW50PVwiYm90dG9tQ2VudGVyXCIgYXJyb3c+XHJcbiAgICAgIDxCdXR0b25cclxuICAgICAgICB0eXBlPVwibGlua1wiXHJcbiAgICAgICAgc3R5bGU9e3sgY29sb3I6ICd3aGl0ZScsIGZvbnRWYXJpYW50OiAnYWxsLXNtYWxsLWNhcHMnIH19PlxyXG4gICAgICAgIHttb2RlfVxyXG4gICAgPC9CdXR0b24+XHJcbiAgICA8L0Ryb3Bkb3duPlxyXG4gIClcclxuXHJcbn0iXSwic291cmNlUm9vdCI6IiJ9