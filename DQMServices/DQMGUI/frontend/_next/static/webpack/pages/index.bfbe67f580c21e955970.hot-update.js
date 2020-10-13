webpackHotUpdate_N_E("pages/index",{

/***/ "./components/browsing/datasetsBrowsing/datasetNameBuilder.tsx":
false,

/***/ "./components/browsing/datasetsBrowsing/partBrowser.tsx":
false,

/***/ "./components/browsing/index.tsx":
false,

/***/ "./components/browsing/lumesectionBroweser.tsx":
false,

/***/ "./components/menu.tsx":
false,

/***/ "./components/navigation/archive_mode_header.tsx":
false,

/***/ "./components/navigation/composedSearch.tsx":
false,

/***/ "./components/navigation/freeSearchResultModal.tsx":
false,

/***/ "./components/navigation/liveModeHeader.tsx":
false,

/***/ "./components/navigation/selectedData.tsx":
false,

/***/ "./components/plots/plot/plotSearch/index.tsx":
false,

/***/ "./components/runInfo/index.tsx":
false,

/***/ "./components/runInfo/runInfoModal.tsx":
false,

/***/ "./components/runInfo/runStartTimeStamp.tsx":
false,

/***/ "./components/workspaces/index.tsx":
false,

/***/ "./containers/display/header.tsx":
false,

/***/ "./hooks/useAvailbleAndNotAvailableDatasetPartsOptions.tsx":
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
/* harmony import */ var next_head__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! next/head */ "./node_modules/next/dist/next-server/lib/head.js");
/* harmony import */ var next_head__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(next_head__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../styles/styledComponents */ "./styles/styledComponents.ts");
/* harmony import */ var _utils_pages__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../utils/pages */ "./utils/pages/index.tsx");
/* harmony import */ var _containers_display_content_constent_switching__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../containers/display/content/constent_switching */ "./containers/display/content/constent_switching.tsx");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/pages/index.tsx",
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
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Tooltip"], {
    title: "Back to main page",
    placement: "bottomLeft",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 39,
      columnNumber: 11
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLogoDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 40,
      columnNumber: 13
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLogoWrapper"], {
    onClick: function onClick(e) {
      return Object(_utils_pages__WEBPACK_IMPORTED_MODULE_5__["backToMainPage"])(e);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 41,
      columnNumber: 15
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLogo"], {
    src: "./images/CMSlogo_white_red_nolabel_1024_May2014.png",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 42,
      columnNumber: 17
    }
  }))))), __jsx(_containers_display_content_constent_switching__WEBPACK_IMPORTED_MODULE_6__["ContentSwitching"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 51,
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

/***/ }),

/***/ "./workspaces/online.ts":
false

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vcGFnZXMvaW5kZXgudHN4Il0sIm5hbWVzIjpbIkluZGV4Iiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJpc0RhdGFzZXRBbmRSdW5OdW1iZXJTZWxlY3RlZCIsInJ1bl9udW1iZXIiLCJkYXRhc2V0X25hbWUiLCJlIiwiYmFja1RvTWFpblBhZ2UiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFFQTtBQUNBO0FBQ0E7QUFFQTtBQVNBO0FBRUE7O0FBRUEsSUFBTUEsS0FBZ0MsR0FBRyxTQUFuQ0EsS0FBbUMsR0FBTTtBQUFBOztBQUM3QztBQUNBLE1BQU1DLE1BQU0sR0FBR0MsNkRBQVMsRUFBeEI7QUFDQSxNQUFNQyxLQUFpQixHQUFHRixNQUFNLENBQUNFLEtBQWpDO0FBRUEsTUFBTUMsNkJBQTZCLEdBQ2pDLENBQUMsQ0FBQ0QsS0FBSyxDQUFDRSxVQUFSLElBQXNCLENBQUMsQ0FBQ0YsS0FBSyxDQUFDRyxZQURoQztBQUdBLFNBQ0UsTUFBQyxrRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxnREFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0U7QUFDRSxlQUFXLEVBQUMsV0FEZDtBQUVFLFFBQUksRUFBQyxpQkFGUDtBQUdFLE9BQUcsRUFBQyxxREFITjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FERixFQVFFLE1BQUMscUVBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMscUVBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsNENBQUQ7QUFBUyxTQUFLLEVBQUMsbUJBQWY7QUFBbUMsYUFBUyxFQUFDLFlBQTdDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHNFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDBFQUFEO0FBQW1CLFdBQU8sRUFBRSxpQkFBQ0MsQ0FBRDtBQUFBLGFBQUtDLG1FQUFjLENBQUNELENBQUQsQ0FBbkI7QUFBQSxLQUE1QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxtRUFBRDtBQUFZLE9BQUcsRUFBQyxxREFBaEI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBREYsQ0FERixDQURGLENBREYsRUFjRSxNQUFDLCtGQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFkRixDQVJGLENBREY7QUEyQkQsQ0FuQ0Q7O0dBQU1QLEs7VUFFV0UscUQ7OztLQUZYRixLO0FBcUNTQSxvRUFBZiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC5iZmJlNjdmNTgwYzIxZTk1NTk3MC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IE5leHRQYWdlIH0gZnJvbSAnbmV4dCc7XG5pbXBvcnQgSGVhZCBmcm9tICduZXh0L2hlYWQnO1xuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xuaW1wb3J0IHsgVG9vbHRpcCB9IGZyb20gJ2FudGQnO1xuXG5pbXBvcnQge1xuICBTdHlsZWRIZWFkZXIsXG4gIFN0eWxlZExheW91dCxcbiAgU3R5bGVkRGl2LFxuICBTdHlsZWRMb2dvV3JhcHBlcixcbiAgU3R5bGVkTG9nbyxcbiAgU3R5bGVkTG9nb0Rpdixcbn0gZnJvbSAnLi4vc3R5bGVzL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgRm9sZGVyUGF0aFF1ZXJ5LCBRdWVyeVByb3BzIH0gZnJvbSAnLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgYmFja1RvTWFpblBhZ2UgfSBmcm9tICcuLi91dGlscy9wYWdlcyc7XG5pbXBvcnQgeyBIZWFkZXIgfSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvaGVhZGVyJztcbmltcG9ydCB7IENvbnRlbnRTd2l0Y2hpbmcgfSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvY29udGVudC9jb25zdGVudF9zd2l0Y2hpbmcnO1xuXG5jb25zdCBJbmRleDogTmV4dFBhZ2U8Rm9sZGVyUGF0aFF1ZXJ5PiA9ICgpID0+IHtcbiAgLy8gV2UgZ3JhYiB0aGUgcXVlcnkgZnJvbSB0aGUgVVJMOlxuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XG5cbiAgY29uc3QgaXNEYXRhc2V0QW5kUnVuTnVtYmVyU2VsZWN0ZWQgPVxuICAgICEhcXVlcnkucnVuX251bWJlciAmJiAhIXF1ZXJ5LmRhdGFzZXRfbmFtZTtcblxuICByZXR1cm4gKFxuICAgIDxTdHlsZWREaXY+XG4gICAgICA8SGVhZD5cbiAgICAgICAgPHNjcmlwdFxuICAgICAgICAgIGNyb3NzT3JpZ2luPVwiYW5vbnltb3VzXCJcbiAgICAgICAgICB0eXBlPVwidGV4dC9qYXZhc2NyaXB0XCJcbiAgICAgICAgICBzcmM9XCIuL2pzcm9vdC01LjguMC9zY3JpcHRzL0pTUm9vdENvcmUuanM/MmQmaGlzdCZtb3JlMmRcIlxuICAgICAgICA+PC9zY3JpcHQ+XG4gICAgICA8L0hlYWQ+XG4gICAgICA8U3R5bGVkTGF5b3V0PlxuICAgICAgICA8U3R5bGVkSGVhZGVyPlxuICAgICAgICAgIDxUb29sdGlwIHRpdGxlPVwiQmFjayB0byBtYWluIHBhZ2VcIiBwbGFjZW1lbnQ9XCJib3R0b21MZWZ0XCI+XG4gICAgICAgICAgICA8U3R5bGVkTG9nb0Rpdj5cbiAgICAgICAgICAgICAgPFN0eWxlZExvZ29XcmFwcGVyIG9uQ2xpY2s9eyhlKT0+YmFja1RvTWFpblBhZ2UoZSl9PlxuICAgICAgICAgICAgICAgIDxTdHlsZWRMb2dvIHNyYz1cIi4vaW1hZ2VzL0NNU2xvZ29fd2hpdGVfcmVkX25vbGFiZWxfMTAyNF9NYXkyMDE0LnBuZ1wiIC8+XG4gICAgICAgICAgICAgIDwvU3R5bGVkTG9nb1dyYXBwZXI+XG4gICAgICAgICAgICA8L1N0eWxlZExvZ29EaXY+XG4gICAgICAgICAgPC9Ub29sdGlwPlxuICAgICAgICAgIHsvKiA8SGVhZGVyXG4gICAgICAgICAgICBpc0RhdGFzZXRBbmRSdW5OdW1iZXJTZWxlY3RlZD17aXNEYXRhc2V0QW5kUnVuTnVtYmVyU2VsZWN0ZWR9XG4gICAgICAgICAgICBxdWVyeT17cXVlcnl9XG4gICAgICAgICAgLz4gKi99XG4gICAgICAgIDwvU3R5bGVkSGVhZGVyPlxuICAgICAgICA8Q29udGVudFN3aXRjaGluZyAvPlxuICAgICAgPC9TdHlsZWRMYXlvdXQ+XG4gICAgPC9TdHlsZWREaXY+XG4gICk7XG59O1xuXG5leHBvcnQgZGVmYXVsdCBJbmRleDtcbiJdLCJzb3VyY2VSb290IjoiIn0=