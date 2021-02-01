webpackHotUpdate_N_E("pages/index",{

/***/ "./node_modules/antd/lib/layout/layout.js":
false,

/***/ "./pages/index.tsx":
/*!*************************!*\
  !*** ./pages/index.tsx ***!
  \*************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var next_head__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! next/head */ "./node_modules/next/head.js");
/* harmony import */ var next_head__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(next_head__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/router */ "./node_modules/next/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../styles/styledComponents */ "./styles/styledComponents.ts");
/* harmony import */ var _utils_pages__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../utils/pages */ "./utils/pages/index.tsx");
/* harmony import */ var _containers_display_header__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../containers/display/header */ "./containers/display/header.tsx");
/* harmony import */ var _containers_display_content_constent_switching__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../containers/display/content/constent_switching */ "./containers/display/content/constent_switching.tsx");
/* harmony import */ var _components_modes_modesSelection__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../components/modes/modesSelection */ "./components/modes/modesSelection.tsx");
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/pages/index.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0___default.a.createElement;










var Index = function Index() {
  _s();

  // We grab the query from the URL:
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"])();
  var query = router.query;
  var isDatasetAndRunNumberSelected = !!query.run_number && !!query.dataset_name;
  return __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 29,
      columnNumber: 5
    }
  }, __jsx(next_head__WEBPACK_IMPORTED_MODULE_1___default.a, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 30,
      columnNumber: 7
    }
  }, __jsx("script", {
    crossOrigin: "anonymous",
    type: "text/javascript",
    src: "./jsroot-5.8.0/scripts/JSRootCore.js?2d&hist&more2d",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 31,
      columnNumber: 9
    }
  })), __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLayout"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 37,
      columnNumber: 7
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledHeader"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 38,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 39,
      columnNumber: 11
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Col"], {
    style: {
      display: 'flex',
      alignItems: 'center'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 40,
      columnNumber: 13
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Tooltip"], {
    title: "Back to main page",
    placement: "bottomLeft",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 41,
      columnNumber: 15
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLogoDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 42,
      columnNumber: 17
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLogoWrapper"], {
    onClick: function onClick(e) {
      return Object(_utils_pages__WEBPACK_IMPORTED_MODULE_5__["backToMainPage"])(e);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 43,
      columnNumber: 19
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLogo"], {
    src: "./images/CMSlogo_white_red_nolabel_1024_May2014.png",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 44,
      columnNumber: 21
    }
  })))), __jsx(_components_modes_modesSelection__WEBPACK_IMPORTED_MODULE_8__["ModesSelection"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 48,
      columnNumber: 14
    }
  }))), __jsx(_containers_display_header__WEBPACK_IMPORTED_MODULE_6__["Header"], {
    isDatasetAndRunNumberSelected: isDatasetAndRunNumberSelected,
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 51,
      columnNumber: 11
    }
  })), __jsx(_containers_display_content_constent_switching__WEBPACK_IMPORTED_MODULE_7__["ContentSwitching"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 56,
      columnNumber: 9
    }
  })));
};

_s(Index, "fN7XvhJ+p5oE6+Xlo0NJmXpxjC8=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"]];
});

_c = Index;
/* harmony default export */ __webpack_exports__["default"] = (Index);

var _c;

$RefreshReg$(_c, "Index");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vcGFnZXMvaW5kZXgudHN4Il0sIm5hbWVzIjpbIkluZGV4Iiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJpc0RhdGFzZXRBbmRSdW5OdW1iZXJTZWxlY3RlZCIsInJ1bl9udW1iZXIiLCJkYXRhc2V0X25hbWUiLCJkaXNwbGF5IiwiYWxpZ25JdGVtcyIsImUiLCJiYWNrVG9NYWluUGFnZSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBRUE7QUFDQTtBQUNBO0FBRUE7QUFTQTtBQUNBO0FBQ0E7QUFDQTs7QUFFQSxJQUFNQSxLQUFnQyxHQUFHLFNBQW5DQSxLQUFtQyxHQUFNO0FBQUE7O0FBQzdDO0FBQ0EsTUFBTUMsTUFBTSxHQUFHQyw2REFBUyxFQUF4QjtBQUNBLE1BQU1DLEtBQWlCLEdBQUdGLE1BQU0sQ0FBQ0UsS0FBakM7QUFDQSxNQUFNQyw2QkFBNkIsR0FDakMsQ0FBQyxDQUFDRCxLQUFLLENBQUNFLFVBQVIsSUFBc0IsQ0FBQyxDQUFDRixLQUFLLENBQUNHLFlBRGhDO0FBR0EsU0FDRSxNQUFDLGtFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLGdEQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRTtBQUNFLGVBQVcsRUFBQyxXQURkO0FBRUUsUUFBSSxFQUFDLGlCQUZQO0FBR0UsT0FBRyxFQUFDLHFEQUhOO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQURGLEVBUUUsTUFBQyxxRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxxRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyx3Q0FBRDtBQUFLLFNBQUssRUFBRTtBQUFFQyxhQUFPLEVBQUUsTUFBWDtBQUFtQkMsZ0JBQVUsRUFBRTtBQUEvQixLQUFaO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDRDQUFEO0FBQVMsU0FBSyxFQUFDLG1CQUFmO0FBQW1DLGFBQVMsRUFBQyxZQUE3QztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxzRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywwRUFBRDtBQUFtQixXQUFPLEVBQUUsaUJBQUNDLENBQUQ7QUFBQSxhQUFPQyxtRUFBYyxDQUFDRCxDQUFELENBQXJCO0FBQUEsS0FBNUI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsbUVBQUQ7QUFBWSxPQUFHLEVBQUMscURBQWhCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQURGLENBREYsQ0FERixFQVFDLE1BQUMsK0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQVJELENBREYsQ0FERixFQWFFLE1BQUMsaUVBQUQ7QUFDRSxpQ0FBNkIsRUFBRUwsNkJBRGpDO0FBRUUsU0FBSyxFQUFFRCxLQUZUO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFiRixDQURGLEVBbUJFLE1BQUMsK0ZBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQW5CRixDQVJGLENBREY7QUFnQ0QsQ0F2Q0Q7O0dBQU1ILEs7VUFFV0UscUQ7OztLQUZYRixLO0FBeUNTQSxvRUFBZiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC5iNTgzNWIyZmM2MGQ1OTAwZTdiNy5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgTmV4dFBhZ2UgfSBmcm9tICduZXh0JztcclxuaW1wb3J0IEhlYWQgZnJvbSAnbmV4dC9oZWFkJztcclxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xyXG5pbXBvcnQgeyBDb2wsIFRvb2x0aXAgfSBmcm9tICdhbnRkJztcclxuXHJcbmltcG9ydCB7XHJcbiAgU3R5bGVkSGVhZGVyLFxyXG4gIFN0eWxlZExheW91dCxcclxuICBTdHlsZWREaXYsXHJcbiAgU3R5bGVkTG9nb1dyYXBwZXIsXHJcbiAgU3R5bGVkTG9nbyxcclxuICBTdHlsZWRMb2dvRGl2LFxyXG59IGZyb20gJy4uL3N0eWxlcy9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHsgRm9sZGVyUGF0aFF1ZXJ5LCBRdWVyeVByb3BzIH0gZnJvbSAnLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyBiYWNrVG9NYWluUGFnZSB9IGZyb20gJy4uL3V0aWxzL3BhZ2VzJztcclxuaW1wb3J0IHsgSGVhZGVyIH0gZnJvbSAnLi4vY29udGFpbmVycy9kaXNwbGF5L2hlYWRlcic7XHJcbmltcG9ydCB7IENvbnRlbnRTd2l0Y2hpbmcgfSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvY29udGVudC9jb25zdGVudF9zd2l0Y2hpbmcnO1xyXG5pbXBvcnQgeyBNb2Rlc1NlbGVjdGlvbiB9IGZyb20gJy4uL2NvbXBvbmVudHMvbW9kZXMvbW9kZXNTZWxlY3Rpb24nO1xyXG5cclxuY29uc3QgSW5kZXg6IE5leHRQYWdlPEZvbGRlclBhdGhRdWVyeT4gPSAoKSA9PiB7XHJcbiAgLy8gV2UgZ3JhYiB0aGUgcXVlcnkgZnJvbSB0aGUgVVJMOlxyXG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xyXG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xyXG4gIGNvbnN0IGlzRGF0YXNldEFuZFJ1bk51bWJlclNlbGVjdGVkID1cclxuICAgICEhcXVlcnkucnVuX251bWJlciAmJiAhIXF1ZXJ5LmRhdGFzZXRfbmFtZTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxTdHlsZWREaXY+XHJcbiAgICAgIDxIZWFkPlxyXG4gICAgICAgIDxzY3JpcHRcclxuICAgICAgICAgIGNyb3NzT3JpZ2luPVwiYW5vbnltb3VzXCJcclxuICAgICAgICAgIHR5cGU9XCJ0ZXh0L2phdmFzY3JpcHRcIlxyXG4gICAgICAgICAgc3JjPVwiLi9qc3Jvb3QtNS44LjAvc2NyaXB0cy9KU1Jvb3RDb3JlLmpzPzJkJmhpc3QmbW9yZTJkXCJcclxuICAgICAgICA+PC9zY3JpcHQ+XHJcbiAgICAgIDwvSGVhZD5cclxuICAgICAgPFN0eWxlZExheW91dD5cclxuICAgICAgICA8U3R5bGVkSGVhZGVyPlxyXG4gICAgICAgICAgPENvbD5cclxuICAgICAgICAgICAgPENvbCBzdHlsZT17eyBkaXNwbGF5OiAnZmxleCcsIGFsaWduSXRlbXM6ICdjZW50ZXInIH19PlxyXG4gICAgICAgICAgICAgIDxUb29sdGlwIHRpdGxlPVwiQmFjayB0byBtYWluIHBhZ2VcIiBwbGFjZW1lbnQ9XCJib3R0b21MZWZ0XCI+XHJcbiAgICAgICAgICAgICAgICA8U3R5bGVkTG9nb0Rpdj5cclxuICAgICAgICAgICAgICAgICAgPFN0eWxlZExvZ29XcmFwcGVyIG9uQ2xpY2s9eyhlKSA9PiBiYWNrVG9NYWluUGFnZShlKX0+XHJcbiAgICAgICAgICAgICAgICAgICAgPFN0eWxlZExvZ28gc3JjPVwiLi9pbWFnZXMvQ01TbG9nb193aGl0ZV9yZWRfbm9sYWJlbF8xMDI0X01heTIwMTQucG5nXCIgLz5cclxuICAgICAgICAgICAgICAgICAgPC9TdHlsZWRMb2dvV3JhcHBlcj5cclxuICAgICAgICAgICAgICAgIDwvU3R5bGVkTG9nb0Rpdj5cclxuICAgICAgICAgICAgICA8L1Rvb2x0aXA+XHJcbiAgICAgICAgICAgICA8TW9kZXNTZWxlY3Rpb24gLz5cclxuICAgICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICA8L0NvbD5cclxuICAgICAgICAgIDxIZWFkZXJcclxuICAgICAgICAgICAgaXNEYXRhc2V0QW5kUnVuTnVtYmVyU2VsZWN0ZWQ9e2lzRGF0YXNldEFuZFJ1bk51bWJlclNlbGVjdGVkfVxyXG4gICAgICAgICAgICBxdWVyeT17cXVlcnl9XHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgIDwvU3R5bGVkSGVhZGVyPlxyXG4gICAgICAgIDxDb250ZW50U3dpdGNoaW5nIC8+XHJcbiAgICAgIDwvU3R5bGVkTGF5b3V0PlxyXG4gICAgPC9TdHlsZWREaXY+XHJcbiAgKTtcclxufTtcclxuXHJcbmV4cG9ydCBkZWZhdWx0IEluZGV4O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9