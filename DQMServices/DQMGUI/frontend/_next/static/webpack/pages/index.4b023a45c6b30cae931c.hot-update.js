webpackHotUpdate_N_E("pages/index",{

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
/* harmony import */ var antd_lib_layout_layout__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! antd/lib/layout/layout */ "./node_modules/antd/lib/layout/layout.js");
/* harmony import */ var antd_lib_layout_layout__WEBPACK_IMPORTED_MODULE_9___default = /*#__PURE__*/__webpack_require__.n(antd_lib_layout_layout__WEBPACK_IMPORTED_MODULE_9__);
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
      lineNumber: 31,
      columnNumber: 5
    }
  }, __jsx(next_head__WEBPACK_IMPORTED_MODULE_1___default.a, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 32,
      columnNumber: 7
    }
  }, __jsx("script", {
    crossOrigin: "anonymous",
    type: "text/javascript",
    src: "./jsroot-5.8.0/scripts/JSRootCore.js?2d&hist&more2d",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 33,
      columnNumber: 9
    }
  })), __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLayout"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 39,
      columnNumber: 7
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledHeader"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 40,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 41,
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
      lineNumber: 42,
      columnNumber: 13
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Tooltip"], {
    title: "Back to main page",
    placement: "bottomLeft",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 43,
      columnNumber: 15
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLogoDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 44,
      columnNumber: 17
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLogoWrapper"], {
    onClick: function onClick(e) {
      return Object(_utils_pages__WEBPACK_IMPORTED_MODULE_5__["backToMainPage"])(e);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 45,
      columnNumber: 19
    }
  }, __jsx(_styles_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledLogo"], {
    src: "./images/CMSlogo_white_red_nolabel_1024_May2014.png",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 46,
      columnNumber: 21
    }
  })))), __jsx(_components_modes_modesSelection__WEBPACK_IMPORTED_MODULE_8__["ModesSelection"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 50,
      columnNumber: 14
    }
  }))), __jsx(_containers_display_header__WEBPACK_IMPORTED_MODULE_6__["Header"], {
    isDatasetAndRunNumberSelected: isDatasetAndRunNumberSelected,
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 53,
      columnNumber: 11
    }
  })), __jsx(_containers_display_content_constent_switching__WEBPACK_IMPORTED_MODULE_7__["ContentSwitching"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 58,
      columnNumber: 9
    }
  }), __jsx(antd_lib_layout_layout__WEBPACK_IMPORTED_MODULE_9__["Footer"], {
    style: {
      background: 'pink'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 59,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vcGFnZXMvaW5kZXgudHN4Il0sIm5hbWVzIjpbIkluZGV4Iiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJpc0RhdGFzZXRBbmRSdW5OdW1iZXJTZWxlY3RlZCIsInJ1bl9udW1iZXIiLCJkYXRhc2V0X25hbWUiLCJkaXNwbGF5IiwiYWxpZ25JdGVtcyIsImUiLCJiYWNrVG9NYWluUGFnZSIsImJhY2tncm91bmQiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBRUE7QUFDQTtBQUNBO0FBRUE7QUFTQTtBQUNBO0FBQ0E7QUFDQTtBQUNBOztBQUdBLElBQU1BLEtBQWdDLEdBQUcsU0FBbkNBLEtBQW1DLEdBQU07QUFBQTs7QUFDN0M7QUFDQSxNQUFNQyxNQUFNLEdBQUdDLDZEQUFTLEVBQXhCO0FBQ0EsTUFBTUMsS0FBaUIsR0FBR0YsTUFBTSxDQUFDRSxLQUFqQztBQUNBLE1BQU1DLDZCQUE2QixHQUNqQyxDQUFDLENBQUNELEtBQUssQ0FBQ0UsVUFBUixJQUFzQixDQUFDLENBQUNGLEtBQUssQ0FBQ0csWUFEaEM7QUFHQSxTQUNFLE1BQUMsa0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsZ0RBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFO0FBQ0UsZUFBVyxFQUFDLFdBRGQ7QUFFRSxRQUFJLEVBQUMsaUJBRlA7QUFHRSxPQUFHLEVBQUMscURBSE47QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBREYsRUFRRSxNQUFDLHFFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHFFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHdDQUFEO0FBQUssU0FBSyxFQUFFO0FBQUVDLGFBQU8sRUFBRSxNQUFYO0FBQW1CQyxnQkFBVSxFQUFFO0FBQS9CLEtBQVo7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsNENBQUQ7QUFBUyxTQUFLLEVBQUMsbUJBQWY7QUFBbUMsYUFBUyxFQUFDLFlBQTdDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHNFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDBFQUFEO0FBQW1CLFdBQU8sRUFBRSxpQkFBQ0MsQ0FBRDtBQUFBLGFBQU9DLG1FQUFjLENBQUNELENBQUQsQ0FBckI7QUFBQSxLQUE1QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxtRUFBRDtBQUFZLE9BQUcsRUFBQyxxREFBaEI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBREYsQ0FERixDQURGLEVBUUMsTUFBQywrRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBUkQsQ0FERixDQURGLEVBYUUsTUFBQyxpRUFBRDtBQUNFLGlDQUE2QixFQUFFTCw2QkFEakM7QUFFRSxTQUFLLEVBQUVELEtBRlQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQWJGLENBREYsRUFtQkUsTUFBQywrRkFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBbkJGLEVBb0JFLE1BQUMsNkRBQUQ7QUFBUSxTQUFLLEVBQUU7QUFBQ1EsZ0JBQVUsRUFBRTtBQUFiLEtBQWY7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQXBCRixDQVJGLENBREY7QUFpQ0QsQ0F4Q0Q7O0dBQU1YLEs7VUFFV0UscUQ7OztLQUZYRixLO0FBMENTQSxvRUFBZiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC40YjAyM2E0NWM2YjMwY2FlOTMxYy5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgTmV4dFBhZ2UgfSBmcm9tICduZXh0JztcclxuaW1wb3J0IEhlYWQgZnJvbSAnbmV4dC9oZWFkJztcclxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xyXG5pbXBvcnQgeyBDb2wsIFRvb2x0aXAsIExheW91dCB9IGZyb20gJ2FudGQnO1xyXG5cclxuaW1wb3J0IHtcclxuICBTdHlsZWRIZWFkZXIsXHJcbiAgU3R5bGVkTGF5b3V0LFxyXG4gIFN0eWxlZERpdixcclxuICBTdHlsZWRMb2dvV3JhcHBlcixcclxuICBTdHlsZWRMb2dvLFxyXG4gIFN0eWxlZExvZ29EaXYsXHJcbn0gZnJvbSAnLi4vc3R5bGVzL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyBGb2xkZXJQYXRoUXVlcnksIFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IGJhY2tUb01haW5QYWdlIH0gZnJvbSAnLi4vdXRpbHMvcGFnZXMnO1xyXG5pbXBvcnQgeyBIZWFkZXIgfSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvaGVhZGVyJztcclxuaW1wb3J0IHsgQ29udGVudFN3aXRjaGluZyB9IGZyb20gJy4uL2NvbnRhaW5lcnMvZGlzcGxheS9jb250ZW50L2NvbnN0ZW50X3N3aXRjaGluZyc7XHJcbmltcG9ydCB7IE1vZGVzU2VsZWN0aW9uIH0gZnJvbSAnLi4vY29tcG9uZW50cy9tb2Rlcy9tb2Rlc1NlbGVjdGlvbic7XHJcbmltcG9ydCB7IEZvb3RlciB9IGZyb20gJ2FudGQvbGliL2xheW91dC9sYXlvdXQnO1xyXG5cclxuXHJcbmNvbnN0IEluZGV4OiBOZXh0UGFnZTxGb2xkZXJQYXRoUXVlcnk+ID0gKCkgPT4ge1xyXG4gIC8vIFdlIGdyYWIgdGhlIHF1ZXJ5IGZyb20gdGhlIFVSTDpcclxuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcclxuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcclxuICBjb25zdCBpc0RhdGFzZXRBbmRSdW5OdW1iZXJTZWxlY3RlZCA9XHJcbiAgICAhIXF1ZXJ5LnJ1bl9udW1iZXIgJiYgISFxdWVyeS5kYXRhc2V0X25hbWU7XHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8U3R5bGVkRGl2PlxyXG4gICAgICA8SGVhZD5cclxuICAgICAgICA8c2NyaXB0XHJcbiAgICAgICAgICBjcm9zc09yaWdpbj1cImFub255bW91c1wiXHJcbiAgICAgICAgICB0eXBlPVwidGV4dC9qYXZhc2NyaXB0XCJcclxuICAgICAgICAgIHNyYz1cIi4vanNyb290LTUuOC4wL3NjcmlwdHMvSlNSb290Q29yZS5qcz8yZCZoaXN0Jm1vcmUyZFwiXHJcbiAgICAgICAgPjwvc2NyaXB0PlxyXG4gICAgICA8L0hlYWQ+XHJcbiAgICAgIDxTdHlsZWRMYXlvdXQ+XHJcbiAgICAgICAgPFN0eWxlZEhlYWRlcj5cclxuICAgICAgICAgIDxDb2w+XHJcbiAgICAgICAgICAgIDxDb2wgc3R5bGU9e3sgZGlzcGxheTogJ2ZsZXgnLCBhbGlnbkl0ZW1zOiAnY2VudGVyJyB9fT5cclxuICAgICAgICAgICAgICA8VG9vbHRpcCB0aXRsZT1cIkJhY2sgdG8gbWFpbiBwYWdlXCIgcGxhY2VtZW50PVwiYm90dG9tTGVmdFwiPlxyXG4gICAgICAgICAgICAgICAgPFN0eWxlZExvZ29EaXY+XHJcbiAgICAgICAgICAgICAgICAgIDxTdHlsZWRMb2dvV3JhcHBlciBvbkNsaWNrPXsoZSkgPT4gYmFja1RvTWFpblBhZ2UoZSl9PlxyXG4gICAgICAgICAgICAgICAgICAgIDxTdHlsZWRMb2dvIHNyYz1cIi4vaW1hZ2VzL0NNU2xvZ29fd2hpdGVfcmVkX25vbGFiZWxfMTAyNF9NYXkyMDE0LnBuZ1wiIC8+XHJcbiAgICAgICAgICAgICAgICAgIDwvU3R5bGVkTG9nb1dyYXBwZXI+XHJcbiAgICAgICAgICAgICAgICA8L1N0eWxlZExvZ29EaXY+XHJcbiAgICAgICAgICAgICAgPC9Ub29sdGlwPlxyXG4gICAgICAgICAgICAgPE1vZGVzU2VsZWN0aW9uIC8+XHJcbiAgICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICA8SGVhZGVyXHJcbiAgICAgICAgICAgIGlzRGF0YXNldEFuZFJ1bk51bWJlclNlbGVjdGVkPXtpc0RhdGFzZXRBbmRSdW5OdW1iZXJTZWxlY3RlZH1cclxuICAgICAgICAgICAgcXVlcnk9e3F1ZXJ5fVxyXG4gICAgICAgICAgLz5cclxuICAgICAgICA8L1N0eWxlZEhlYWRlcj5cclxuICAgICAgICA8Q29udGVudFN3aXRjaGluZyAvPlxyXG4gICAgICAgIDxGb290ZXIgc3R5bGU9e3tiYWNrZ3JvdW5kOiAncGluayd9fSAvPlxyXG4gICAgICA8L1N0eWxlZExheW91dD5cclxuICAgIDwvU3R5bGVkRGl2PlxyXG4gICk7XHJcbn07XHJcblxyXG5leHBvcnQgZGVmYXVsdCBJbmRleDtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==