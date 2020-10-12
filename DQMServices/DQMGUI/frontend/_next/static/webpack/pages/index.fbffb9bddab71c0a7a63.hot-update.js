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
      lineNumber: 36,
      columnNumber: 5
    }
  }, isLoading ? __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["SpinnerWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 38,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["Spinner"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 39,
      columnNumber: 11
    }
  })) : __jsx(react__WEBPACK_IMPORTED_MODULE_0___default.a.Fragment, null, results_grouped.length === 0 && !isLoading && errorsList.length === 0 ? __jsx(_noResultsFound__WEBPACK_IMPORTED_MODULE_3__["NoResultsFound"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 46,
      columnNumber: 17
    }
  }) : !isLoading && errorsList.length === 0 ? __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledTable"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 48,
      columnNumber: 17
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledTableHead"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 49,
      columnNumber: 19
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledTableRow"], {
    noHover: true,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 50,
      columnNumber: 21
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledTableDatasetColumn"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 51,
      columnNumber: 23
    }
  }, "Dataset"), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledTableRunColumn"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 52,
      columnNumber: 23
    }
  }, "Runs"))), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["TableBody"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 55,
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
        lineNumber: 57,
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
        lineNumber: 71,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGFpbmVycy9zZWFyY2gvU2VhcmNoUmVzdWx0cy50c3giXSwibmFtZXMiOlsiU2VhcmNoUmVzdWx0cyIsImhhbmRsZXIiLCJyZXN1bHRzX2dyb3VwZWQiLCJpc0xvYWRpbmciLCJlcnJvcnMiLCJlcnJvcnNMaXN0IiwibGVuZ3RoIiwibWFwIiwiaW5kZXgiLCJkYXRhc2V0IiwicnVucyIsImVycm9yIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBRUE7QUFFQTtBQVlBOztBQVVBLElBQU1BLGFBQXlDLEdBQUcsU0FBNUNBLGFBQTRDLE9BSzVDO0FBQUEsTUFKSkMsT0FJSSxRQUpKQSxPQUlJO0FBQUEsTUFISkMsZUFHSSxRQUhKQSxlQUdJO0FBQUEsTUFGSkMsU0FFSSxRQUZKQSxTQUVJO0FBQUEsTUFESkMsTUFDSSxRQURKQSxNQUNJO0FBQ0osTUFBTUMsVUFBVSxHQUFHRCxNQUFNLElBQUlBLE1BQU0sQ0FBQ0UsTUFBUCxHQUFnQixDQUExQixHQUE4QkYsTUFBOUIsR0FBdUMsRUFBMUQ7QUFFQSxTQUNFLE1BQUMsK0RBQUQ7QUFBZSxhQUFTLEVBQUMsUUFBekI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNHRCxTQUFTLEdBQ1IsTUFBQyxnRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyx5REFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FEUSxHQUtOLG1FQUNHRCxlQUFlLENBQUNJLE1BQWhCLEtBQTJCLENBQTNCLElBQ0MsQ0FBQ0gsU0FERixJQUVDRSxVQUFVLENBQUNDLE1BQVgsS0FBc0IsQ0FGdkIsR0FHRyxNQUFDLDhEQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFISCxHQUlLLENBQUNILFNBQUQsSUFBY0UsVUFBVSxDQUFDQyxNQUFYLEtBQXNCLENBQXBDLEdBQ0YsTUFBQyw2REFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxpRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxnRUFBRDtBQUFnQixXQUFPLE1BQXZCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDBFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsZUFERixFQUVFLE1BQUMsc0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxZQUZGLENBREYsQ0FERixFQU9FLE1BQUMsMkRBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNHSixlQUFlLENBQUNLLEdBQWhCLENBQW9CLGlCQUFvQkMsS0FBcEI7QUFBQSxRQUFHQyxPQUFILFNBQUdBLE9BQUg7QUFBQSxRQUFZQyxJQUFaLFNBQVlBLElBQVo7QUFBQSxXQUNuQixNQUFDLCtDQUFEO0FBQ0UsU0FBRyxFQUFFRCxPQURQO0FBRUUsV0FBSyxFQUFFRCxLQUZUO0FBR0UsYUFBTyxFQUFFUCxPQUhYO0FBSUUsYUFBTyxFQUFFUSxPQUpYO0FBS0UsVUFBSSxFQUFFQyxJQUxSO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFEbUI7QUFBQSxHQUFwQixDQURILENBUEYsQ0FERSxHQXFCQSxDQUFDUCxTQUFELElBQ0FFLFVBQVUsQ0FBQ0MsTUFBWCxHQUFvQixDQURwQixJQUVBRCxVQUFVLENBQUNFLEdBQVgsQ0FBZSxVQUFDSSxLQUFEO0FBQUEsV0FDYixNQUFDLDZEQUFEO0FBQWEsU0FBRyxFQUFFQSxLQUFsQjtBQUF5QixhQUFPLEVBQUVBLEtBQWxDO0FBQXlDLFVBQUksRUFBQyxPQUE5QztBQUFzRCxjQUFRLE1BQTlEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFEYTtBQUFBLEdBQWYsQ0E1QlIsQ0FOTixDQURGO0FBMkNELENBbkREOztLQUFNWCxhO0FBb0RTQSw0RUFBZiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC5mYmZmYjliZGRhYjcxYzBhN2E2My5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0LCB7IEZDLCB1c2VDb250ZXh0LCB1c2VFZmZlY3QgfSBmcm9tICdyZWFjdCc7XG5cbmltcG9ydCBSZXN1bHQgZnJvbSAnLi9SZXN1bHQnO1xuXG5pbXBvcnQge1xuICBTdHlsZWRXcmFwcGVyLFxuICBTcGlubmVyLFxuICBTcGlubmVyV3JhcHBlcixcbiAgU3R5bGVkVGFibGVIZWFkLFxuICBTdHlsZWRUYWJsZVJ1bkNvbHVtbixcbiAgU3R5bGVkVGFibGVEYXRhc2V0Q29sdW1uLFxuICBTdHlsZWRUYWJsZVJvdyxcbiAgU3R5bGVkVGFibGUsXG4gIFRhYmxlQm9keSxcbiAgU3R5bGVkQWxlcnQsXG59IGZyb20gJy4vc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyBOb1Jlc3VsdHNGb3VuZCB9IGZyb20gJy4vbm9SZXN1bHRzRm91bmQnO1xuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnO1xuXG5pbnRlcmZhY2UgU2VhcmNoUmVzdWx0c0ludGVyZmFjZSB7XG4gIHJlc3VsdHNfZ3JvdXBlZDogYW55W107XG4gIGlzTG9hZGluZzogYm9vbGVhbjtcbiAgaGFuZGxlcihydW46IHN0cmluZywgZGF0YXNldDogc3RyaW5nKTogYW55O1xuICBlcnJvcnM/OiBzdHJpbmdbXTtcbn1cblxuY29uc3QgU2VhcmNoUmVzdWx0czogRkM8U2VhcmNoUmVzdWx0c0ludGVyZmFjZT4gPSAoe1xuICBoYW5kbGVyLFxuICByZXN1bHRzX2dyb3VwZWQsXG4gIGlzTG9hZGluZyxcbiAgZXJyb3JzLFxufSkgPT4ge1xuICBjb25zdCBlcnJvcnNMaXN0ID0gZXJyb3JzICYmIGVycm9ycy5sZW5ndGggPiAwID8gZXJyb3JzIDogW107XG4gIFxuICByZXR1cm4gKFxuICAgIDxTdHlsZWRXcmFwcGVyIG92ZXJmbG93eD1cImhpZGRlblwiPlxuICAgICAge2lzTG9hZGluZyA/IChcbiAgICAgICAgPFNwaW5uZXJXcmFwcGVyPlxuICAgICAgICAgIDxTcGlubmVyIC8+XG4gICAgICAgIDwvU3Bpbm5lcldyYXBwZXI+XG4gICAgICApIDogKFxuICAgICAgICAgIDw+XG4gICAgICAgICAgICB7cmVzdWx0c19ncm91cGVkLmxlbmd0aCA9PT0gMCAmJlxuICAgICAgICAgICAgICAhaXNMb2FkaW5nICYmXG4gICAgICAgICAgICAgIGVycm9yc0xpc3QubGVuZ3RoID09PSAwID8gKFxuICAgICAgICAgICAgICAgIDxOb1Jlc3VsdHNGb3VuZCAvPlxuICAgICAgICAgICAgICApIDogIWlzTG9hZGluZyAmJiBlcnJvcnNMaXN0Lmxlbmd0aCA9PT0gMCA/IChcbiAgICAgICAgICAgICAgICA8U3R5bGVkVGFibGU+XG4gICAgICAgICAgICAgICAgICA8U3R5bGVkVGFibGVIZWFkPlxuICAgICAgICAgICAgICAgICAgICA8U3R5bGVkVGFibGVSb3cgbm9Ib3Zlcj5cbiAgICAgICAgICAgICAgICAgICAgICA8U3R5bGVkVGFibGVEYXRhc2V0Q29sdW1uPkRhdGFzZXQ8L1N0eWxlZFRhYmxlRGF0YXNldENvbHVtbj5cbiAgICAgICAgICAgICAgICAgICAgICA8U3R5bGVkVGFibGVSdW5Db2x1bW4+UnVuczwvU3R5bGVkVGFibGVSdW5Db2x1bW4+XG4gICAgICAgICAgICAgICAgICAgIDwvU3R5bGVkVGFibGVSb3c+XG4gICAgICAgICAgICAgICAgICA8L1N0eWxlZFRhYmxlSGVhZD5cbiAgICAgICAgICAgICAgICAgIDxUYWJsZUJvZHk+XG4gICAgICAgICAgICAgICAgICAgIHtyZXN1bHRzX2dyb3VwZWQubWFwKCh7IGRhdGFzZXQsIHJ1bnMgfSwgaW5kZXgpID0+IChcbiAgICAgICAgICAgICAgICAgICAgICA8UmVzdWx0XG4gICAgICAgICAgICAgICAgICAgICAgICBrZXk9e2RhdGFzZXR9XG4gICAgICAgICAgICAgICAgICAgICAgICBpbmRleD17aW5kZXh9XG4gICAgICAgICAgICAgICAgICAgICAgICBoYW5kbGVyPXtoYW5kbGVyfVxuICAgICAgICAgICAgICAgICAgICAgICAgZGF0YXNldD17ZGF0YXNldH1cbiAgICAgICAgICAgICAgICAgICAgICAgIHJ1bnM9e3J1bnN9XG4gICAgICAgICAgICAgICAgICAgICAgLz5cbiAgICAgICAgICAgICAgICAgICAgKSl9XG4gICAgICAgICAgICAgICAgICA8L1RhYmxlQm9keT5cbiAgICAgICAgICAgICAgICA8L1N0eWxlZFRhYmxlPlxuICAgICAgICAgICAgICApIDogKFxuICAgICAgICAgICAgICAgICAgIWlzTG9hZGluZyAmJlxuICAgICAgICAgICAgICAgICAgZXJyb3JzTGlzdC5sZW5ndGggPiAwICYmXG4gICAgICAgICAgICAgICAgICBlcnJvcnNMaXN0Lm1hcCgoZXJyb3IpID0+IChcbiAgICAgICAgICAgICAgICAgICAgPFN0eWxlZEFsZXJ0IGtleT17ZXJyb3J9IG1lc3NhZ2U9e2Vycm9yfSB0eXBlPVwiZXJyb3JcIiBzaG93SWNvbiAvPlxuICAgICAgICAgICAgICAgICAgKSlcbiAgICAgICAgICAgICAgICApfVxuICAgICAgICAgIDwvPlxuICAgICAgICApfVxuICAgIDwvU3R5bGVkV3JhcHBlcj5cbiAgKTtcbn07XG5leHBvcnQgZGVmYXVsdCBTZWFyY2hSZXN1bHRzO1xuIl0sInNvdXJjZVJvb3QiOiIifQ==