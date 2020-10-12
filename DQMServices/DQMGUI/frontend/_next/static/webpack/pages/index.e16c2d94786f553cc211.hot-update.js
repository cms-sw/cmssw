webpackHotUpdate_N_E("pages/index",{

/***/ "./containers/search/SearchResults.tsx":
/*!*********************************************!*\
  !*** ./containers/search/SearchResults.tsx ***!
  \*********************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _Result__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./Result */ "./containers/search/Result.tsx");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./styledComponents */ "./containers/search/styledComponents.tsx");
/* harmony import */ var _noResultsFound__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ./noResultsFound */ "./containers/search/noResultsFound.tsx");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/containers/search/SearchResults.tsx";

var __jsx = react__WEBPACK_IMPORTED_MODULE_0___default.a.createElement;





var SearchResults = function SearchResults(_ref) {
  var handler = _ref.handler,
      results_grouped = _ref.results_grouped,
      isLoading = _ref.isLoading,
      errors = _ref.errors;
  var errorsList = errors && errors.length > 0 ? errors : [];
  console.log(isLoading);
  return __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledWrapper"], {
    overflowx: "hidden",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 35,
      columnNumber: 5
    }
  }, isLoading ? __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["SpinnerWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 37,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["Spinner"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 38,
      columnNumber: 11
    }
  })) : __jsx(react__WEBPACK_IMPORTED_MODULE_0___default.a.Fragment, null, results_grouped.length === 0 && !isLoading && errorsList.length === 0 ? __jsx(_noResultsFound__WEBPACK_IMPORTED_MODULE_3__["NoResultsFound"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 45,
      columnNumber: 17
    }
  }) : !isLoading && errorsList.length === 0 ? __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledTable"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 47,
      columnNumber: 17
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledTableHead"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 48,
      columnNumber: 19
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledTableRow"], {
    noHover: true,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 49,
      columnNumber: 21
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledTableDatasetColumn"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 50,
      columnNumber: 23
    }
  }, "Dataset"), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledTableRunColumn"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 51,
      columnNumber: 23
    }
  }, "Runs"))), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["TableBody"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 54,
      columnNumber: 19
    }
  }, results_grouped.map(function (_ref2, index) {
    var dataset = _ref2.dataset,
        runs = _ref2.runs;
    return __jsx(_Result__WEBPACK_IMPORTED_MODULE_1__["default"], {
      key: dataset,
      index: index,
      handler: handler,
      dataset: dataset,
      runs: runs,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 56,
        columnNumber: 23
      }
    });
  }))) : !isLoading && errorsList.length > 0 && errorsList.map(function (error) {
    return __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledAlert"], {
      key: error,
      message: error,
      type: "error",
      showIcon: true,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 70,
        columnNumber: 21
      }
    });
  })));
};

_c = SearchResults;
/* harmony default export */ __webpack_exports__["default"] = (SearchResults);

var _c;

$RefreshReg$(_c, "SearchResults");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGFpbmVycy9zZWFyY2gvU2VhcmNoUmVzdWx0cy50c3giXSwibmFtZXMiOlsiU2VhcmNoUmVzdWx0cyIsImhhbmRsZXIiLCJyZXN1bHRzX2dyb3VwZWQiLCJpc0xvYWRpbmciLCJlcnJvcnMiLCJlcnJvcnNMaXN0IiwibGVuZ3RoIiwiY29uc29sZSIsImxvZyIsIm1hcCIsImluZGV4IiwiZGF0YXNldCIsInJ1bnMiLCJlcnJvciJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUVBO0FBRUE7QUFZQTs7QUFTQSxJQUFNQSxhQUF5QyxHQUFHLFNBQTVDQSxhQUE0QyxPQUs1QztBQUFBLE1BSkpDLE9BSUksUUFKSkEsT0FJSTtBQUFBLE1BSEpDLGVBR0ksUUFISkEsZUFHSTtBQUFBLE1BRkpDLFNBRUksUUFGSkEsU0FFSTtBQUFBLE1BREpDLE1BQ0ksUUFESkEsTUFDSTtBQUNKLE1BQU1DLFVBQVUsR0FBR0QsTUFBTSxJQUFJQSxNQUFNLENBQUNFLE1BQVAsR0FBZ0IsQ0FBMUIsR0FBOEJGLE1BQTlCLEdBQXVDLEVBQTFEO0FBQ0FHLFNBQU8sQ0FBQ0MsR0FBUixDQUFZTCxTQUFaO0FBQ0EsU0FDRSxNQUFDLCtEQUFEO0FBQWUsYUFBUyxFQUFDLFFBQXpCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDR0EsU0FBUyxHQUNSLE1BQUMsZ0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMseURBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBRFEsR0FLTixtRUFDR0QsZUFBZSxDQUFDSSxNQUFoQixLQUEyQixDQUEzQixJQUNDLENBQUNILFNBREYsSUFFQ0UsVUFBVSxDQUFDQyxNQUFYLEtBQXNCLENBRnZCLEdBR0csTUFBQyw4REFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBSEgsR0FJSyxDQUFDSCxTQUFELElBQWNFLFVBQVUsQ0FBQ0MsTUFBWCxLQUFzQixDQUFwQyxHQUNGLE1BQUMsNkRBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsaUVBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsZ0VBQUQ7QUFBZ0IsV0FBTyxNQUF2QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywwRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLGVBREYsRUFFRSxNQUFDLHNFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsWUFGRixDQURGLENBREYsRUFPRSxNQUFDLDJEQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDR0osZUFBZSxDQUFDTyxHQUFoQixDQUFvQixpQkFBb0JDLEtBQXBCO0FBQUEsUUFBR0MsT0FBSCxTQUFHQSxPQUFIO0FBQUEsUUFBWUMsSUFBWixTQUFZQSxJQUFaO0FBQUEsV0FDbkIsTUFBQywrQ0FBRDtBQUNFLFNBQUcsRUFBRUQsT0FEUDtBQUVFLFdBQUssRUFBRUQsS0FGVDtBQUdFLGFBQU8sRUFBRVQsT0FIWDtBQUlFLGFBQU8sRUFBRVUsT0FKWDtBQUtFLFVBQUksRUFBRUMsSUFMUjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BRG1CO0FBQUEsR0FBcEIsQ0FESCxDQVBGLENBREUsR0FxQkEsQ0FBQ1QsU0FBRCxJQUNBRSxVQUFVLENBQUNDLE1BQVgsR0FBb0IsQ0FEcEIsSUFFQUQsVUFBVSxDQUFDSSxHQUFYLENBQWUsVUFBQ0ksS0FBRDtBQUFBLFdBQ2IsTUFBQyw2REFBRDtBQUFhLFNBQUcsRUFBRUEsS0FBbEI7QUFBeUIsYUFBTyxFQUFFQSxLQUFsQztBQUF5QyxVQUFJLEVBQUMsT0FBOUM7QUFBc0QsY0FBUSxNQUE5RDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BRGE7QUFBQSxHQUFmLENBNUJSLENBTk4sQ0FERjtBQTJDRCxDQW5ERDs7S0FBTWIsYTtBQW9EU0EsNEVBQWYiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguZTE2YzJkOTQ3ODZmNTUzY2MyMTEuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCwgeyBGQywgdXNlQ29udGV4dCwgdXNlRWZmZWN0IH0gZnJvbSAncmVhY3QnO1xuXG5pbXBvcnQgUmVzdWx0IGZyb20gJy4vUmVzdWx0JztcblxuaW1wb3J0IHtcbiAgU3R5bGVkV3JhcHBlcixcbiAgU3Bpbm5lcixcbiAgU3Bpbm5lcldyYXBwZXIsXG4gIFN0eWxlZFRhYmxlSGVhZCxcbiAgU3R5bGVkVGFibGVSdW5Db2x1bW4sXG4gIFN0eWxlZFRhYmxlRGF0YXNldENvbHVtbixcbiAgU3R5bGVkVGFibGVSb3csXG4gIFN0eWxlZFRhYmxlLFxuICBUYWJsZUJvZHksXG4gIFN0eWxlZEFsZXJ0LFxufSBmcm9tICcuL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgTm9SZXN1bHRzRm91bmQgfSBmcm9tICcuL25vUmVzdWx0c0ZvdW5kJztcblxuaW50ZXJmYWNlIFNlYXJjaFJlc3VsdHNJbnRlcmZhY2Uge1xuICByZXN1bHRzX2dyb3VwZWQ6IGFueVtdO1xuICBpc0xvYWRpbmc6IGJvb2xlYW47XG4gIGhhbmRsZXIocnVuOiBzdHJpbmcsIGRhdGFzZXQ6IHN0cmluZyk6IGFueTtcbiAgZXJyb3JzPzogc3RyaW5nW107XG59XG5cbmNvbnN0IFNlYXJjaFJlc3VsdHM6IEZDPFNlYXJjaFJlc3VsdHNJbnRlcmZhY2U+ID0gKHtcbiAgaGFuZGxlcixcbiAgcmVzdWx0c19ncm91cGVkLFxuICBpc0xvYWRpbmcsXG4gIGVycm9ycyxcbn0pID0+IHtcbiAgY29uc3QgZXJyb3JzTGlzdCA9IGVycm9ycyAmJiBlcnJvcnMubGVuZ3RoID4gMCA/IGVycm9ycyA6IFtdO1xuICBjb25zb2xlLmxvZyhpc0xvYWRpbmcpXG4gIHJldHVybiAoXG4gICAgPFN0eWxlZFdyYXBwZXIgb3ZlcmZsb3d4PVwiaGlkZGVuXCI+XG4gICAgICB7aXNMb2FkaW5nID8gKFxuICAgICAgICA8U3Bpbm5lcldyYXBwZXI+XG4gICAgICAgICAgPFNwaW5uZXIgLz5cbiAgICAgICAgPC9TcGlubmVyV3JhcHBlcj5cbiAgICAgICkgOiAoXG4gICAgICAgICAgPD5cbiAgICAgICAgICAgIHtyZXN1bHRzX2dyb3VwZWQubGVuZ3RoID09PSAwICYmXG4gICAgICAgICAgICAgICFpc0xvYWRpbmcgJiZcbiAgICAgICAgICAgICAgZXJyb3JzTGlzdC5sZW5ndGggPT09IDAgPyAoXG4gICAgICAgICAgICAgICAgPE5vUmVzdWx0c0ZvdW5kIC8+XG4gICAgICAgICAgICAgICkgOiAhaXNMb2FkaW5nICYmIGVycm9yc0xpc3QubGVuZ3RoID09PSAwID8gKFxuICAgICAgICAgICAgICAgIDxTdHlsZWRUYWJsZT5cbiAgICAgICAgICAgICAgICAgIDxTdHlsZWRUYWJsZUhlYWQ+XG4gICAgICAgICAgICAgICAgICAgIDxTdHlsZWRUYWJsZVJvdyBub0hvdmVyPlxuICAgICAgICAgICAgICAgICAgICAgIDxTdHlsZWRUYWJsZURhdGFzZXRDb2x1bW4+RGF0YXNldDwvU3R5bGVkVGFibGVEYXRhc2V0Q29sdW1uPlxuICAgICAgICAgICAgICAgICAgICAgIDxTdHlsZWRUYWJsZVJ1bkNvbHVtbj5SdW5zPC9TdHlsZWRUYWJsZVJ1bkNvbHVtbj5cbiAgICAgICAgICAgICAgICAgICAgPC9TdHlsZWRUYWJsZVJvdz5cbiAgICAgICAgICAgICAgICAgIDwvU3R5bGVkVGFibGVIZWFkPlxuICAgICAgICAgICAgICAgICAgPFRhYmxlQm9keT5cbiAgICAgICAgICAgICAgICAgICAge3Jlc3VsdHNfZ3JvdXBlZC5tYXAoKHsgZGF0YXNldCwgcnVucyB9LCBpbmRleCkgPT4gKFxuICAgICAgICAgICAgICAgICAgICAgIDxSZXN1bHRcbiAgICAgICAgICAgICAgICAgICAgICAgIGtleT17ZGF0YXNldH1cbiAgICAgICAgICAgICAgICAgICAgICAgIGluZGV4PXtpbmRleH1cbiAgICAgICAgICAgICAgICAgICAgICAgIGhhbmRsZXI9e2hhbmRsZXJ9XG4gICAgICAgICAgICAgICAgICAgICAgICBkYXRhc2V0PXtkYXRhc2V0fVxuICAgICAgICAgICAgICAgICAgICAgICAgcnVucz17cnVuc31cbiAgICAgICAgICAgICAgICAgICAgICAvPlxuICAgICAgICAgICAgICAgICAgICApKX1cbiAgICAgICAgICAgICAgICAgIDwvVGFibGVCb2R5PlxuICAgICAgICAgICAgICAgIDwvU3R5bGVkVGFibGU+XG4gICAgICAgICAgICAgICkgOiAoXG4gICAgICAgICAgICAgICAgICAhaXNMb2FkaW5nICYmXG4gICAgICAgICAgICAgICAgICBlcnJvcnNMaXN0Lmxlbmd0aCA+IDAgJiZcbiAgICAgICAgICAgICAgICAgIGVycm9yc0xpc3QubWFwKChlcnJvcikgPT4gKFxuICAgICAgICAgICAgICAgICAgICA8U3R5bGVkQWxlcnQga2V5PXtlcnJvcn0gbWVzc2FnZT17ZXJyb3J9IHR5cGU9XCJlcnJvclwiIHNob3dJY29uIC8+XG4gICAgICAgICAgICAgICAgICApKVxuICAgICAgICAgICAgICAgICl9XG4gICAgICAgICAgPC8+XG4gICAgICAgICl9XG4gICAgPC9TdHlsZWRXcmFwcGVyPlxuICApO1xufTtcbmV4cG9ydCBkZWZhdWx0IFNlYXJjaFJlc3VsdHM7XG4iXSwic291cmNlUm9vdCI6IiJ9