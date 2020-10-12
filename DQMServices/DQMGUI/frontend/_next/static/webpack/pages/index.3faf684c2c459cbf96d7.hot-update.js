webpackHotUpdate_N_E("pages/index",{

/***/ "./containers/display/content/display_folders_or_plots.tsx":
/*!*****************************************************************!*\
  !*** ./containers/display/content/display_folders_or_plots.tsx ***!
  \*****************************************************************/
/*! exports provided: DisplayFordersOrPlots */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "DisplayFordersOrPlots", function() { return DisplayFordersOrPlots; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../styledComponents */ "./containers/display/styledComponents.tsx");
/* harmony import */ var _search_styledComponents__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../search/styledComponents */ "./containers/search/styledComponents.tsx");
/* harmony import */ var _search_noResultsFound__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../search/noResultsFound */ "./containers/search/noResultsFound.tsx");
/* harmony import */ var _components_styledComponents__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../../components/styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _directories__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./directories */ "./containers/display/content/directories.tsx");
/* harmony import */ var _components_plots_plot__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../../components/plots/plot */ "./components/plots/plot/index.tsx");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/containers/display/content/display_folders_or_plots.tsx";

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];








var DisplayFordersOrPlots = function DisplayFordersOrPlots(_ref) {
  var plots = _ref.plots,
      selected_plots = _ref.selected_plots,
      plots_grouped_by_layouts = _ref.plots_grouped_by_layouts,
      isLoading = _ref.isLoading,
      viewPlotsPosition = _ref.viewPlotsPosition,
      proportion = _ref.proportion,
      errors = _ref.errors,
      filteredFolders = _ref.filteredFolders;
  return __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_2__["Wrapper"], {
    any_selected_plots: selected_plots.length > 0 && errors.length === 0,
    position: viewPlotsPosition,
    proportion: proportion,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 44,
      columnNumber: 5
    }
  }, isLoading && filteredFolders.length === 0 ? __jsx(_search_styledComponents__WEBPACK_IMPORTED_MODULE_3__["SpinnerWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 50,
      columnNumber: 9
    }
  }, __jsx(_search_styledComponents__WEBPACK_IMPORTED_MODULE_3__["Spinner"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 51,
      columnNumber: 11
    }
  })) : __jsx(react__WEBPACK_IMPORTED_MODULE_0__["Fragment"], null, !isLoading && filteredFolders.length === 0 && plots.length === 0 && errors.length === 0 ? __jsx(_search_noResultsFound__WEBPACK_IMPORTED_MODULE_4__["NoResultsFound"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 59,
      columnNumber: 13
    }
  }) : errors.length === 0 ? __jsx(react__WEBPACK_IMPORTED_MODULE_0__["Fragment"], null, __jsx(_components_styledComponents__WEBPACK_IMPORTED_MODULE_5__["CustomRow"], {
    width: "100%",
    space: '2',
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 62,
      columnNumber: 15
    }
  }, __jsx(_directories__WEBPACK_IMPORTED_MODULE_6__["Directories"], {
    directories: filteredFolders,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 63,
      columnNumber: 17
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Row"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 65,
      columnNumber: 15
    }
  }, __jsx(_components_plots_plot__WEBPACK_IMPORTED_MODULE_7__["LeftSidePlots"], {
    plots: plots,
    selected_plots: selected_plots,
    plots_grouped_by_layouts: plots_grouped_by_layouts,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 66,
      columnNumber: 17
    }
  }))) : !isLoading && errors.length > 0 && errors.map(function (error) {
    return __jsx(_search_styledComponents__WEBPACK_IMPORTED_MODULE_3__["StyledAlert"], {
      key: error,
      message: error,
      type: "error",
      showIcon: true,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 77,
        columnNumber: 15
      }
    });
  })));
};
_c = DisplayFordersOrPlots;

var _c;

$RefreshReg$(_c, "DisplayFordersOrPlots");

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGFpbmVycy9kaXNwbGF5L2NvbnRlbnQvZGlzcGxheV9mb2xkZXJzX29yX3Bsb3RzLnRzeCJdLCJuYW1lcyI6WyJEaXNwbGF5Rm9yZGVyc09yUGxvdHMiLCJwbG90cyIsInNlbGVjdGVkX3Bsb3RzIiwicGxvdHNfZ3JvdXBlZF9ieV9sYXlvdXRzIiwiaXNMb2FkaW5nIiwidmlld1Bsb3RzUG9zaXRpb24iLCJwcm9wb3J0aW9uIiwiZXJyb3JzIiwiZmlsdGVyZWRGb2xkZXJzIiwibGVuZ3RoIiwibWFwIiwiZXJyb3IiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBRUE7QUFDQTtBQUtBO0FBQ0E7QUFDQTtBQUNBO0FBb0JPLElBQU1BLHFCQUFxQixHQUFHLFNBQXhCQSxxQkFBd0IsT0FTakI7QUFBQSxNQVJsQkMsS0FRa0IsUUFSbEJBLEtBUWtCO0FBQUEsTUFQbEJDLGNBT2tCLFFBUGxCQSxjQU9rQjtBQUFBLE1BTmxCQyx3QkFNa0IsUUFObEJBLHdCQU1rQjtBQUFBLE1BTGxCQyxTQUtrQixRQUxsQkEsU0FLa0I7QUFBQSxNQUpsQkMsaUJBSWtCLFFBSmxCQSxpQkFJa0I7QUFBQSxNQUhsQkMsVUFHa0IsUUFIbEJBLFVBR2tCO0FBQUEsTUFGbEJDLE1BRWtCLFFBRmxCQSxNQUVrQjtBQUFBLE1BRGxCQyxlQUNrQixRQURsQkEsZUFDa0I7QUFDbEIsU0FDRSxNQUFDLHlEQUFEO0FBQ0Usc0JBQWtCLEVBQUVOLGNBQWMsQ0FBQ08sTUFBZixHQUF3QixDQUF4QixJQUE2QkYsTUFBTSxDQUFDRSxNQUFQLEtBQWtCLENBRHJFO0FBRUUsWUFBUSxFQUFFSixpQkFGWjtBQUdFLGNBQVUsRUFBRUMsVUFIZDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBS0dGLFNBQVMsSUFBS0ksZUFBZSxDQUFDQyxNQUFoQixLQUEyQixDQUF6QyxHQUNDLE1BQUMsdUVBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsZ0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBREQsR0FLQyw0REFDRyxDQUFDTCxTQUFELElBQ0RJLGVBQWUsQ0FBQ0MsTUFBaEIsS0FBMkIsQ0FEMUIsSUFFRFIsS0FBSyxDQUFDUSxNQUFOLEtBQWlCLENBRmhCLElBR0RGLE1BQU0sQ0FBQ0UsTUFBUCxLQUFrQixDQUhqQixHQUlDLE1BQUMscUVBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQUpELEdBS0dGLE1BQU0sQ0FBQ0UsTUFBUCxLQUFrQixDQUFsQixHQUNGLDREQUNFLE1BQUMsc0VBQUQ7QUFBVyxTQUFLLEVBQUMsTUFBakI7QUFBd0IsU0FBSyxFQUFFLEdBQS9CO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHdEQUFEO0FBQWEsZUFBVyxFQUFFRCxlQUExQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FERixFQUlFLE1BQUMsd0NBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsb0VBQUQ7QUFDRSxTQUFLLEVBQUVQLEtBRFQ7QUFFRSxrQkFBYyxFQUFFQyxjQUZsQjtBQUdFLDRCQUF3QixFQUFFQyx3QkFINUI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLENBSkYsQ0FERSxHQWNGLENBQUNDLFNBQUQsSUFDQUcsTUFBTSxDQUFDRSxNQUFQLEdBQWdCLENBRGhCLElBRUFGLE1BQU0sQ0FBQ0csR0FBUCxDQUFXLFVBQUNDLEtBQUQ7QUFBQSxXQUNULE1BQUMsb0VBQUQ7QUFBYSxTQUFHLEVBQUVBLEtBQWxCO0FBQXlCLGFBQU8sRUFBRUEsS0FBbEM7QUFBeUMsVUFBSSxFQUFDLE9BQTlDO0FBQXNELGNBQVEsTUFBOUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQURTO0FBQUEsR0FBWCxDQXRCSixDQVZKLENBREY7QUF5Q0QsQ0FuRE07S0FBTVgscUIiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguM2ZhZjY4NGMyYzQ1OWNiZjk2ZDcuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IFJvdyB9IGZyb20gJ2FudGQnO1xuXG5pbXBvcnQgeyBXcmFwcGVyIH0gZnJvbSAnLi4vc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQge1xuICBTcGlubmVyV3JhcHBlcixcbiAgU3Bpbm5lcixcbiAgU3R5bGVkQWxlcnQsXG59IGZyb20gJy4uLy4uL3NlYXJjaC9zdHlsZWRDb21wb25lbnRzJztcbmltcG9ydCB7IE5vUmVzdWx0c0ZvdW5kIH0gZnJvbSAnLi4vLi4vc2VhcmNoL25vUmVzdWx0c0ZvdW5kJztcbmltcG9ydCB7IEN1c3RvbVJvdyB9IGZyb20gJy4uLy4uLy4uL2NvbXBvbmVudHMvc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyBEaXJlY3RvcmllcyB9IGZyb20gJy4vZGlyZWN0b3JpZXMnO1xuaW1wb3J0IHsgTGVmdFNpZGVQbG90cyB9IGZyb20gJy4uLy4uLy4uL2NvbXBvbmVudHMvcGxvdHMvcGxvdCc7XG5pbXBvcnQge1xuICBQbG90RGF0YVByb3BzLFxuICBQbG90c0dyb3VwZWRCeUxheW91dHNJbnRlcmZhY2UsXG4gIE9wdGlvblByb3BzLFxuICBRdWVyeVByb3BzLFxufSBmcm9tICcuLi9pbnRlcmZhY2VzJztcblxuaW50ZXJmYWNlIENvbnRlbnRQcm9wcyB7XG4gIHBsb3RzOiBQbG90RGF0YVByb3BzW107XG4gIHNlbGVjdGVkX3Bsb3RzOiBhbnlbXTtcbiAgcGxvdHNfZ3JvdXBlZF9ieV9sYXlvdXRzPzogUGxvdHNHcm91cGVkQnlMYXlvdXRzSW50ZXJmYWNlO1xuICBpc0xvYWRpbmc6IGJvb2xlYW47XG4gIHZpZXdQbG90c1Bvc2l0aW9uOiBPcHRpb25Qcm9wcztcbiAgcHJvcG9ydGlvbjogT3B0aW9uUHJvcHM7XG4gIGVycm9yczogc3RyaW5nW107XG4gIGZpbHRlcmVkRm9sZGVyczogYW55W107XG4gIHF1ZXJ5OiBRdWVyeVByb3BzO1xufVxuXG5leHBvcnQgY29uc3QgRGlzcGxheUZvcmRlcnNPclBsb3RzID0gKHtcbiAgcGxvdHMsXG4gIHNlbGVjdGVkX3Bsb3RzLFxuICBwbG90c19ncm91cGVkX2J5X2xheW91dHMsXG4gIGlzTG9hZGluZyxcbiAgdmlld1Bsb3RzUG9zaXRpb24sXG4gIHByb3BvcnRpb24sXG4gIGVycm9ycyxcbiAgZmlsdGVyZWRGb2xkZXJzLFxufTogQ29udGVudFByb3BzKSA9PiB7XG4gIHJldHVybiAoXG4gICAgPFdyYXBwZXJcbiAgICAgIGFueV9zZWxlY3RlZF9wbG90cz17c2VsZWN0ZWRfcGxvdHMubGVuZ3RoID4gMCAmJiBlcnJvcnMubGVuZ3RoID09PSAwfVxuICAgICAgcG9zaXRpb249e3ZpZXdQbG90c1Bvc2l0aW9ufVxuICAgICAgcHJvcG9ydGlvbj17cHJvcG9ydGlvbn1cbiAgICA+XG4gICAgICB7aXNMb2FkaW5nICAmJiBmaWx0ZXJlZEZvbGRlcnMubGVuZ3RoID09PSAwPyAoXG4gICAgICAgIDxTcGlubmVyV3JhcHBlcj5cbiAgICAgICAgICA8U3Bpbm5lciAvPlxuICAgICAgICA8L1NwaW5uZXJXcmFwcGVyPlxuICAgICAgKSA6IChcbiAgICAgICAgPD5cbiAgICAgICAgICB7IWlzTG9hZGluZyAmJlxuICAgICAgICAgIGZpbHRlcmVkRm9sZGVycy5sZW5ndGggPT09IDAgJiZcbiAgICAgICAgICBwbG90cy5sZW5ndGggPT09IDAgJiZcbiAgICAgICAgICBlcnJvcnMubGVuZ3RoID09PSAwID8gKFxuICAgICAgICAgICAgPE5vUmVzdWx0c0ZvdW5kIC8+XG4gICAgICAgICAgKSA6IGVycm9ycy5sZW5ndGggPT09IDAgPyAoXG4gICAgICAgICAgICA8PlxuICAgICAgICAgICAgICA8Q3VzdG9tUm93IHdpZHRoPVwiMTAwJVwiIHNwYWNlPXsnMid9PlxuICAgICAgICAgICAgICAgIDxEaXJlY3RvcmllcyBkaXJlY3Rvcmllcz17ZmlsdGVyZWRGb2xkZXJzfSAvPlxuICAgICAgICAgICAgICA8L0N1c3RvbVJvdz5cbiAgICAgICAgICAgICAgPFJvdz5cbiAgICAgICAgICAgICAgICA8TGVmdFNpZGVQbG90c1xuICAgICAgICAgICAgICAgICAgcGxvdHM9e3Bsb3RzfVxuICAgICAgICAgICAgICAgICAgc2VsZWN0ZWRfcGxvdHM9e3NlbGVjdGVkX3Bsb3RzfVxuICAgICAgICAgICAgICAgICAgcGxvdHNfZ3JvdXBlZF9ieV9sYXlvdXRzPXtwbG90c19ncm91cGVkX2J5X2xheW91dHN9XG4gICAgICAgICAgICAgICAgLz5cbiAgICAgICAgICAgICAgPC9Sb3c+XG4gICAgICAgICAgICA8Lz5cbiAgICAgICAgICApIDogKFxuICAgICAgICAgICAgIWlzTG9hZGluZyAmJlxuICAgICAgICAgICAgZXJyb3JzLmxlbmd0aCA+IDAgJiZcbiAgICAgICAgICAgIGVycm9ycy5tYXAoKGVycm9yKSA9PiAoXG4gICAgICAgICAgICAgIDxTdHlsZWRBbGVydCBrZXk9e2Vycm9yfSBtZXNzYWdlPXtlcnJvcn0gdHlwZT1cImVycm9yXCIgc2hvd0ljb24gLz5cbiAgICAgICAgICAgICkpXG4gICAgICAgICAgKX1cbiAgICAgICAgPC8+XG4gICAgICApfVxuICAgIDwvV3JhcHBlcj5cbiAgKTtcbn07XG4iXSwic291cmNlUm9vdCI6IiJ9