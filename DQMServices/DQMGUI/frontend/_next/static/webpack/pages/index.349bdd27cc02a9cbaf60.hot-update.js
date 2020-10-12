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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGFpbmVycy9zZWFyY2gvU2VhcmNoUmVzdWx0cy50c3giXSwibmFtZXMiOlsiU2VhcmNoUmVzdWx0cyIsImhhbmRsZXIiLCJyZXN1bHRzX2dyb3VwZWQiLCJpc0xvYWRpbmciLCJlcnJvcnMiLCJlcnJvcnNMaXN0IiwibGVuZ3RoIiwibWFwIiwiaW5kZXgiLCJkYXRhc2V0IiwicnVucyIsImVycm9yIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBRUE7QUFFQTtBQVlBOztBQVNBLElBQU1BLGFBQXlDLEdBQUcsU0FBNUNBLGFBQTRDLE9BSzVDO0FBQUEsTUFKSkMsT0FJSSxRQUpKQSxPQUlJO0FBQUEsTUFISkMsZUFHSSxRQUhKQSxlQUdJO0FBQUEsTUFGSkMsU0FFSSxRQUZKQSxTQUVJO0FBQUEsTUFESkMsTUFDSSxRQURKQSxNQUNJO0FBQ0osTUFBTUMsVUFBVSxHQUFHRCxNQUFNLElBQUlBLE1BQU0sQ0FBQ0UsTUFBUCxHQUFnQixDQUExQixHQUE4QkYsTUFBOUIsR0FBdUMsRUFBMUQ7QUFFQSxTQUNFLE1BQUMsK0RBQUQ7QUFBZSxhQUFTLEVBQUMsUUFBekI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNHRCxTQUFTLEdBQ1IsTUFBQyxnRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyx5REFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FEUSxHQUtOLG1FQUNHRCxlQUFlLENBQUNJLE1BQWhCLEtBQTJCLENBQTNCLElBQ0MsQ0FBQ0gsU0FERixJQUVDRSxVQUFVLENBQUNDLE1BQVgsS0FBc0IsQ0FGdkIsR0FHRyxNQUFDLDhEQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFISCxHQUlLLENBQUNILFNBQUQsSUFBY0UsVUFBVSxDQUFDQyxNQUFYLEtBQXNCLENBQXBDLEdBQ0YsTUFBQyw2REFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxpRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxnRUFBRDtBQUFnQixXQUFPLE1BQXZCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDBFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsZUFERixFQUVFLE1BQUMsc0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxZQUZGLENBREYsQ0FERixFQU9FLE1BQUMsMkRBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNHSixlQUFlLENBQUNLLEdBQWhCLENBQW9CLGlCQUFvQkMsS0FBcEI7QUFBQSxRQUFHQyxPQUFILFNBQUdBLE9BQUg7QUFBQSxRQUFZQyxJQUFaLFNBQVlBLElBQVo7QUFBQSxXQUNuQixNQUFDLCtDQUFEO0FBQ0UsU0FBRyxFQUFFRCxPQURQO0FBRUUsV0FBSyxFQUFFRCxLQUZUO0FBR0UsYUFBTyxFQUFFUCxPQUhYO0FBSUUsYUFBTyxFQUFFUSxPQUpYO0FBS0UsVUFBSSxFQUFFQyxJQUxSO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFEbUI7QUFBQSxHQUFwQixDQURILENBUEYsQ0FERSxHQXFCQSxDQUFDUCxTQUFELElBQ0FFLFVBQVUsQ0FBQ0MsTUFBWCxHQUFvQixDQURwQixJQUVBRCxVQUFVLENBQUNFLEdBQVgsQ0FBZSxVQUFDSSxLQUFEO0FBQUEsV0FDYixNQUFDLDZEQUFEO0FBQWEsU0FBRyxFQUFFQSxLQUFsQjtBQUF5QixhQUFPLEVBQUVBLEtBQWxDO0FBQXlDLFVBQUksRUFBQyxPQUE5QztBQUFzRCxjQUFRLE1BQTlEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFEYTtBQUFBLEdBQWYsQ0E1QlIsQ0FOTixDQURGO0FBMkNELENBbkREOztLQUFNWCxhO0FBb0RTQSw0RUFBZiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4zNDliZGQyN2NjMDJhOWNiYWY2MC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0LCB7IEZDLCB1c2VDb250ZXh0LCB1c2VFZmZlY3QgfSBmcm9tICdyZWFjdCc7XG5cbmltcG9ydCBSZXN1bHQgZnJvbSAnLi9SZXN1bHQnO1xuXG5pbXBvcnQge1xuICBTdHlsZWRXcmFwcGVyLFxuICBTcGlubmVyLFxuICBTcGlubmVyV3JhcHBlcixcbiAgU3R5bGVkVGFibGVIZWFkLFxuICBTdHlsZWRUYWJsZVJ1bkNvbHVtbixcbiAgU3R5bGVkVGFibGVEYXRhc2V0Q29sdW1uLFxuICBTdHlsZWRUYWJsZVJvdyxcbiAgU3R5bGVkVGFibGUsXG4gIFRhYmxlQm9keSxcbiAgU3R5bGVkQWxlcnQsXG59IGZyb20gJy4vc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyBOb1Jlc3VsdHNGb3VuZCB9IGZyb20gJy4vbm9SZXN1bHRzRm91bmQnO1xuXG5pbnRlcmZhY2UgU2VhcmNoUmVzdWx0c0ludGVyZmFjZSB7XG4gIHJlc3VsdHNfZ3JvdXBlZDogYW55W107XG4gIGlzTG9hZGluZzogYm9vbGVhbjtcbiAgaGFuZGxlcihydW46IHN0cmluZywgZGF0YXNldDogc3RyaW5nKTogYW55O1xuICBlcnJvcnM/OiBzdHJpbmdbXTtcbn1cblxuY29uc3QgU2VhcmNoUmVzdWx0czogRkM8U2VhcmNoUmVzdWx0c0ludGVyZmFjZT4gPSAoe1xuICBoYW5kbGVyLFxuICByZXN1bHRzX2dyb3VwZWQsXG4gIGlzTG9hZGluZyxcbiAgZXJyb3JzLFxufSkgPT4ge1xuICBjb25zdCBlcnJvcnNMaXN0ID0gZXJyb3JzICYmIGVycm9ycy5sZW5ndGggPiAwID8gZXJyb3JzIDogW107XG5cbiAgcmV0dXJuIChcbiAgICA8U3R5bGVkV3JhcHBlciBvdmVyZmxvd3g9XCJoaWRkZW5cIj5cbiAgICAgIHtpc0xvYWRpbmcgPyAoXG4gICAgICAgIDxTcGlubmVyV3JhcHBlcj5cbiAgICAgICAgICA8U3Bpbm5lciAvPlxuICAgICAgICA8L1NwaW5uZXJXcmFwcGVyPlxuICAgICAgKSA6IChcbiAgICAgICAgICA8PlxuICAgICAgICAgICAge3Jlc3VsdHNfZ3JvdXBlZC5sZW5ndGggPT09IDAgJiZcbiAgICAgICAgICAgICAgIWlzTG9hZGluZyAmJlxuICAgICAgICAgICAgICBlcnJvcnNMaXN0Lmxlbmd0aCA9PT0gMCA/IChcbiAgICAgICAgICAgICAgICA8Tm9SZXN1bHRzRm91bmQgLz5cbiAgICAgICAgICAgICAgKSA6ICFpc0xvYWRpbmcgJiYgZXJyb3JzTGlzdC5sZW5ndGggPT09IDAgPyAoXG4gICAgICAgICAgICAgICAgPFN0eWxlZFRhYmxlPlxuICAgICAgICAgICAgICAgICAgPFN0eWxlZFRhYmxlSGVhZD5cbiAgICAgICAgICAgICAgICAgICAgPFN0eWxlZFRhYmxlUm93IG5vSG92ZXI+XG4gICAgICAgICAgICAgICAgICAgICAgPFN0eWxlZFRhYmxlRGF0YXNldENvbHVtbj5EYXRhc2V0PC9TdHlsZWRUYWJsZURhdGFzZXRDb2x1bW4+XG4gICAgICAgICAgICAgICAgICAgICAgPFN0eWxlZFRhYmxlUnVuQ29sdW1uPlJ1bnM8L1N0eWxlZFRhYmxlUnVuQ29sdW1uPlxuICAgICAgICAgICAgICAgICAgICA8L1N0eWxlZFRhYmxlUm93PlxuICAgICAgICAgICAgICAgICAgPC9TdHlsZWRUYWJsZUhlYWQ+XG4gICAgICAgICAgICAgICAgICA8VGFibGVCb2R5PlxuICAgICAgICAgICAgICAgICAgICB7cmVzdWx0c19ncm91cGVkLm1hcCgoeyBkYXRhc2V0LCBydW5zIH0sIGluZGV4KSA9PiAoXG4gICAgICAgICAgICAgICAgICAgICAgPFJlc3VsdFxuICAgICAgICAgICAgICAgICAgICAgICAga2V5PXtkYXRhc2V0fVxuICAgICAgICAgICAgICAgICAgICAgICAgaW5kZXg9e2luZGV4fVxuICAgICAgICAgICAgICAgICAgICAgICAgaGFuZGxlcj17aGFuZGxlcn1cbiAgICAgICAgICAgICAgICAgICAgICAgIGRhdGFzZXQ9e2RhdGFzZXR9XG4gICAgICAgICAgICAgICAgICAgICAgICBydW5zPXtydW5zfVxuICAgICAgICAgICAgICAgICAgICAgIC8+XG4gICAgICAgICAgICAgICAgICAgICkpfVxuICAgICAgICAgICAgICAgICAgPC9UYWJsZUJvZHk+XG4gICAgICAgICAgICAgICAgPC9TdHlsZWRUYWJsZT5cbiAgICAgICAgICAgICAgKSA6IChcbiAgICAgICAgICAgICAgICAgICFpc0xvYWRpbmcgJiZcbiAgICAgICAgICAgICAgICAgIGVycm9yc0xpc3QubGVuZ3RoID4gMCAmJlxuICAgICAgICAgICAgICAgICAgZXJyb3JzTGlzdC5tYXAoKGVycm9yKSA9PiAoXG4gICAgICAgICAgICAgICAgICAgIDxTdHlsZWRBbGVydCBrZXk9e2Vycm9yfSBtZXNzYWdlPXtlcnJvcn0gdHlwZT1cImVycm9yXCIgc2hvd0ljb24gLz5cbiAgICAgICAgICAgICAgICAgICkpXG4gICAgICAgICAgICAgICAgKX1cbiAgICAgICAgICA8Lz5cbiAgICAgICAgKX1cbiAgICA8L1N0eWxlZFdyYXBwZXI+XG4gICk7XG59O1xuZXhwb3J0IGRlZmF1bHQgU2VhcmNoUmVzdWx0cztcbiJdLCJzb3VyY2VSb290IjoiIn0=