webpackHotUpdate_N_E("pages/index",{

/***/ "./components/navigation/composedSearch.tsx":
/*!**************************************************!*\
  !*** ./components/navigation/composedSearch.tsx ***!
  \**************************************************/
/*! exports provided: ComposedSearch */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ComposedSearch", function() { return ComposedSearch; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../containers/display/styledComponents */ "./containers/display/styledComponents.tsx");
/* harmony import */ var _liveModeHeader__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./liveModeHeader */ "./components/navigation/liveModeHeader.tsx");
/* harmony import */ var _archive_mode_header__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./archive_mode_header */ "./components/navigation/archive_mode_header.tsx");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/navigation/composedSearch.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];







var ComposedSearch = function ComposedSearch() {
  _s();

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"])();
  var query = router.query;
  var set_on_live_mode = query.run_number === '0' && query.dataset_name === '/Global/Online/ALL';
  return __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomRow"], {
    width: "100%",
    display: "flex",
    justifycontent: "space-between",
    alignitems: "center",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 21,
      columnNumber: 5
    }
  }, set_on_live_mode ? __jsx(_liveModeHeader__WEBPACK_IMPORTED_MODULE_5__["LiveModeHeader"], {
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 28,
      columnNumber: 9
    }
  }) : __jsx(_archive_mode_header__WEBPACK_IMPORTED_MODULE_6__["ArchiveModeHeader"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 30,
      columnNumber: 9
    }
  }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_4__["WrapperDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 32,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 33,
      columnNumber: 9
    }
  }), __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 36,
      columnNumber: 9
    }
  })));
};

_s(ComposedSearch, "fN7XvhJ+p5oE6+Xlo0NJmXpxjC8=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"]];
});

_c = ComposedSearch;

var _c;

$RefreshReg$(_c, "ComposedSearch");

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

/***/ }),

/***/ "./components/workspaces/index.tsx":
false,

/***/ "./workspaces/online.ts":
false

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2NvbXBvc2VkU2VhcmNoLnRzeCJdLCJuYW1lcyI6WyJDb21wb3NlZFNlYXJjaCIsInJvdXRlciIsInVzZVJvdXRlciIsInF1ZXJ5Iiwic2V0X29uX2xpdmVfbW9kZSIsInJ1bl9udW1iZXIiLCJkYXRhc2V0X25hbWUiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBR0E7QUFHQTtBQUNBO0FBQ0E7QUFFTyxJQUFNQSxjQUFjLEdBQUcsU0FBakJBLGNBQWlCLEdBQU07QUFBQTs7QUFDbEMsTUFBTUMsTUFBTSxHQUFHQyw2REFBUyxFQUF4QjtBQUNBLE1BQU1DLEtBQWlCLEdBQUdGLE1BQU0sQ0FBQ0UsS0FBakM7QUFFQSxNQUFNQyxnQkFBZ0IsR0FDcEJELEtBQUssQ0FBQ0UsVUFBTixLQUFxQixHQUFyQixJQUE0QkYsS0FBSyxDQUFDRyxZQUFOLEtBQXVCLG9CQURyRDtBQUdBLFNBQ0UsTUFBQywyREFBRDtBQUNFLFNBQUssRUFBQyxNQURSO0FBRUUsV0FBTyxFQUFDLE1BRlY7QUFHRSxrQkFBYyxFQUFDLGVBSGpCO0FBSUUsY0FBVSxFQUFDLFFBSmI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQU1HRixnQkFBZ0IsR0FDZixNQUFDLDhEQUFEO0FBQWdCLFNBQUssRUFBRUQsS0FBdkI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURlLEdBR2YsTUFBQyxzRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBVEosRUFXRSxNQUFDLCtFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixFQUlFLE1BQUMsd0NBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQUpGLENBWEYsQ0FERjtBQXNCRCxDQTdCTTs7R0FBTUgsYztVQUNJRSxxRDs7O0tBREpGLGMiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguMzZiMzA5MGU2YzhiZjkyNDRmMjIuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IENvbCB9IGZyb20gJ2FudGQnO1xuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xuXG5pbXBvcnQgV29ya3NwYWNlcyBmcm9tICcuLi93b3Jrc3BhY2VzJztcbmltcG9ydCB7IEN1c3RvbVJvdyB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgUGxvdFNlYXJjaCB9IGZyb20gJy4uL3Bsb3RzL3Bsb3QvcGxvdFNlYXJjaCc7XG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgV3JhcHBlckRpdiB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9zdHlsZWRDb21wb25lbnRzJztcbmltcG9ydCB7IExpdmVNb2RlSGVhZGVyIH0gZnJvbSAnLi9saXZlTW9kZUhlYWRlcic7XG5pbXBvcnQgeyBBcmNoaXZlTW9kZUhlYWRlciB9IGZyb20gJy4vYXJjaGl2ZV9tb2RlX2hlYWRlcic7XG5cbmV4cG9ydCBjb25zdCBDb21wb3NlZFNlYXJjaCA9ICgpID0+IHtcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xuXG4gIGNvbnN0IHNldF9vbl9saXZlX21vZGUgPVxuICAgIHF1ZXJ5LnJ1bl9udW1iZXIgPT09ICcwJyAmJiBxdWVyeS5kYXRhc2V0X25hbWUgPT09ICcvR2xvYmFsL09ubGluZS9BTEwnO1xuXG4gIHJldHVybiAoXG4gICAgPEN1c3RvbVJvd1xuICAgICAgd2lkdGg9XCIxMDAlXCJcbiAgICAgIGRpc3BsYXk9XCJmbGV4XCJcbiAgICAgIGp1c3RpZnljb250ZW50PVwic3BhY2UtYmV0d2VlblwiXG4gICAgICBhbGlnbml0ZW1zPVwiY2VudGVyXCJcbiAgICA+XG4gICAgICB7c2V0X29uX2xpdmVfbW9kZSA/IChcbiAgICAgICAgPExpdmVNb2RlSGVhZGVyIHF1ZXJ5PXtxdWVyeX0gLz5cbiAgICAgICkgOiAoXG4gICAgICAgIDxBcmNoaXZlTW9kZUhlYWRlciAvPlxuICAgICAgKX1cbiAgICAgIDxXcmFwcGVyRGl2PlxuICAgICAgICA8Q29sPlxuICAgICAgICAgIHsvKiA8V29ya3NwYWNlcyAvPiAqL31cbiAgICAgICAgPC9Db2w+XG4gICAgICAgIDxDb2w+XG4gICAgICAgICAgey8qIDxQbG90U2VhcmNoIGlzTG9hZGluZ0ZvbGRlcnM9e2ZhbHNlfSAvPiAqL31cbiAgICAgICAgPC9Db2w+XG4gICAgICA8L1dyYXBwZXJEaXY+XG4gICAgPC9DdXN0b21Sb3c+XG4gICk7XG59O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==