webpackHotUpdate_N_E("pages/index",{

/***/ "./containers/display/content/constent_switching.tsx":
/*!***********************************************************!*\
  !*** ./containers/display/content/constent_switching.tsx ***!
  \***********************************************************/
/*! exports provided: ContentSwitching */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ContentSwitching", function() { return ContentSwitching; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _folders_and_plots_content__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ./folders_and_plots_content */ "./containers/display/content/folders_and_plots_content.tsx");
/* harmony import */ var _hooks_useSearch__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../../hooks/useSearch */ "./hooks/useSearch.tsx");
/* harmony import */ var _search_SearchResults__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../search/SearchResults */ "./containers/search/SearchResults.tsx");
/* harmony import */ var _search_styledComponents__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../search/styledComponents */ "./containers/search/styledComponents.tsx");
/* harmony import */ var _components_utils__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../../components/utils */ "./components/utils.ts");
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../utils */ "./containers/display/utils.ts");
/* harmony import */ var _workspaces_offline__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../../../workspaces/offline */ "./workspaces/offline.ts");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../../../config/config */ "./config/config.ts");
/* harmony import */ var _components_initialPage_latestRuns__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../../../components/initialPage/latestRuns */ "./components/initialPage/latestRuns.tsx");
/* harmony import */ var _hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ../../../hooks/useUpdateInLiveMode */ "./hooks/useUpdateInLiveMode.tsx");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/containers/display/content/constent_switching.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];












var ContentSwitching = function ContentSwitching() {
  _s();

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_1__["useRouter"])();
  var query = router.query;

  var _useUpdateLiveMode = Object(_hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_11__["useUpdateLiveMode"])(),
      set_update = _useUpdateLiveMode.set_update;

  var workspaceOption = query.workspace ? query.workspace : _workspaces_offline__WEBPACK_IMPORTED_MODULE_8__["workspaces"][0].workspaces[2].label;

  var _useSearch = Object(_hooks_useSearch__WEBPACK_IMPORTED_MODULE_3__["useSearch"])(query.search_run_number, query.search_dataset_name),
      results_grouped = _useSearch.results_grouped,
      searching = _useSearch.searching,
      isLoading = _useSearch.isLoading,
      errors = _useSearch.errors; //serchResultsHandler when you selecting run, dataset from search results


  var serchResultsHandler = function serchResultsHandler(run, dataset) {
    set_update(false);

    var _seperateRunAndLumiIn = Object(_components_utils__WEBPACK_IMPORTED_MODULE_6__["seperateRunAndLumiInSearch"])(run.toString()),
        parsedRun = _seperateRunAndLumiIn.parsedRun,
        parsedLumi = _seperateRunAndLumiIn.parsedLumi;

    Object(_utils__WEBPACK_IMPORTED_MODULE_7__["changeRouter"])(Object(_utils__WEBPACK_IMPORTED_MODULE_7__["getChangedQueryParams"])({
      lumi: parsedLumi,
      run_number: parsedRun,
      dataset_name: dataset,
      workspaces: workspaceOption,
      plot_search: ''
    }, query));
  };

  if (query.dataset_name && query.run_number) {
    return __jsx(_folders_and_plots_content__WEBPACK_IMPORTED_MODULE_2__["default"], {
      run_number: query.run_number || '',
      dataset_name: query.dataset_name || '',
      folder_path: query.folder_path || '',
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 56,
        columnNumber: 7
      }
    });
  } else if (searching) {
    return __jsx(_search_SearchResults__WEBPACK_IMPORTED_MODULE_4__["default"], {
      isLoading: isLoading,
      results_grouped: results_grouped,
      handler: serchResultsHandler,
      errors: errors,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 64,
        columnNumber: 7
      }
    });
  } // !query.dataset_name && !query.run_number because I don't want
  // to see latest runs list, when I'm loading folders or plots
  //  folders and  plots are visible, when dataset_name and run_number is set
  else if (_config_config__WEBPACK_IMPORTED_MODULE_9__["functions_config"].new_back_end.latest_runs && !query.dataset_name && !query.run_number) {
      return __jsx(_components_initialPage_latestRuns__WEBPACK_IMPORTED_MODULE_10__["LatestRuns"], {
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 80,
          columnNumber: 12
        }
      });
    }

  return __jsx(_search_styledComponents__WEBPACK_IMPORTED_MODULE_5__["NotFoundDivWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 83,
      columnNumber: 5
    }
  }, __jsx(_search_styledComponents__WEBPACK_IMPORTED_MODULE_5__["NotFoundDiv"], {
    noBorder: true,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 84,
      columnNumber: 7
    }
  }, __jsx(_search_styledComponents__WEBPACK_IMPORTED_MODULE_5__["ChartIcon"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 85,
      columnNumber: 9
    }
  }), "Welcome to DQM GUI"));
};

_s(ContentSwitching, "s+EPdt8jT4UvGhdwEPMq97yX3I4=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_1__["useRouter"], _hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_11__["useUpdateLiveMode"], _hooks_useSearch__WEBPACK_IMPORTED_MODULE_3__["useSearch"]];
});

_c = ContentSwitching;

var _c;

$RefreshReg$(_c, "ContentSwitching");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGFpbmVycy9kaXNwbGF5L2NvbnRlbnQvY29uc3RlbnRfc3dpdGNoaW5nLnRzeCJdLCJuYW1lcyI6WyJDb250ZW50U3dpdGNoaW5nIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJ1c2VVcGRhdGVMaXZlTW9kZSIsInNldF91cGRhdGUiLCJ3b3Jrc3BhY2VPcHRpb24iLCJ3b3Jrc3BhY2UiLCJ3b3Jrc3BhY2VzIiwibGFiZWwiLCJ1c2VTZWFyY2giLCJzZWFyY2hfcnVuX251bWJlciIsInNlYXJjaF9kYXRhc2V0X25hbWUiLCJyZXN1bHRzX2dyb3VwZWQiLCJzZWFyY2hpbmciLCJpc0xvYWRpbmciLCJlcnJvcnMiLCJzZXJjaFJlc3VsdHNIYW5kbGVyIiwicnVuIiwiZGF0YXNldCIsInNlcGVyYXRlUnVuQW5kTHVtaUluU2VhcmNoIiwidG9TdHJpbmciLCJwYXJzZWRSdW4iLCJwYXJzZWRMdW1pIiwiY2hhbmdlUm91dGVyIiwiZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zIiwibHVtaSIsInJ1bl9udW1iZXIiLCJkYXRhc2V0X25hbWUiLCJwbG90X3NlYXJjaCIsImZvbGRlcl9wYXRoIiwiZnVuY3Rpb25zX2NvbmZpZyIsIm5ld19iYWNrX2VuZCIsImxhdGVzdF9ydW5zIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUdBO0FBQ0E7QUFDQTtBQUNBO0FBS0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBRU8sSUFBTUEsZ0JBQWdCLEdBQUcsU0FBbkJBLGdCQUFtQixHQUFNO0FBQUE7O0FBQ3BDLE1BQU1DLE1BQU0sR0FBR0MsNkRBQVMsRUFBeEI7QUFDQSxNQUFNQyxLQUFpQixHQUFHRixNQUFNLENBQUNFLEtBQWpDOztBQUZvQywyQkFHYkMscUZBQWlCLEVBSEo7QUFBQSxNQUc1QkMsVUFINEIsc0JBRzVCQSxVQUg0Qjs7QUFJcEMsTUFBTUMsZUFBZSxHQUFHSCxLQUFLLENBQUNJLFNBQU4sR0FDcEJKLEtBQUssQ0FBQ0ksU0FEYyxHQUVwQkMsOERBQVUsQ0FBQyxDQUFELENBQVYsQ0FBY0EsVUFBZCxDQUF5QixDQUF6QixFQUE0QkMsS0FGaEM7O0FBSm9DLG1CQVFzQkMsa0VBQVMsQ0FDakVQLEtBQUssQ0FBQ1EsaUJBRDJELEVBRWpFUixLQUFLLENBQUNTLG1CQUYyRCxDQVIvQjtBQUFBLE1BUTVCQyxlQVI0QixjQVE1QkEsZUFSNEI7QUFBQSxNQVFYQyxTQVJXLGNBUVhBLFNBUlc7QUFBQSxNQVFBQyxTQVJBLGNBUUFBLFNBUkE7QUFBQSxNQVFXQyxNQVJYLGNBUVdBLE1BUlgsRUFZcEM7OztBQUNBLE1BQU1DLG1CQUFtQixHQUFHLFNBQXRCQSxtQkFBc0IsQ0FBQ0MsR0FBRCxFQUFjQyxPQUFkLEVBQWtDO0FBQzVEZCxjQUFVLENBQUMsS0FBRCxDQUFWOztBQUQ0RCxnQ0FHMUJlLG9GQUEwQixDQUMxREYsR0FBRyxDQUFDRyxRQUFKLEVBRDBELENBSEE7QUFBQSxRQUdwREMsU0FIb0QseUJBR3BEQSxTQUhvRDtBQUFBLFFBR3pDQyxVQUh5Qyx5QkFHekNBLFVBSHlDOztBQU81REMsK0RBQVksQ0FDVkMsb0VBQXFCLENBQ25CO0FBQ0VDLFVBQUksRUFBRUgsVUFEUjtBQUVFSSxnQkFBVSxFQUFFTCxTQUZkO0FBR0VNLGtCQUFZLEVBQUVULE9BSGhCO0FBSUVYLGdCQUFVLEVBQUVGLGVBSmQ7QUFLRXVCLGlCQUFXLEVBQUU7QUFMZixLQURtQixFQVFuQjFCLEtBUm1CLENBRFgsQ0FBWjtBQVlELEdBbkJEOztBQXFCQSxNQUFJQSxLQUFLLENBQUN5QixZQUFOLElBQXNCekIsS0FBSyxDQUFDd0IsVUFBaEMsRUFBNEM7QUFDMUMsV0FDRSxNQUFDLGtFQUFEO0FBQ0UsZ0JBQVUsRUFBRXhCLEtBQUssQ0FBQ3dCLFVBQU4sSUFBb0IsRUFEbEM7QUFFRSxrQkFBWSxFQUFFeEIsS0FBSyxDQUFDeUIsWUFBTixJQUFzQixFQUZ0QztBQUdFLGlCQUFXLEVBQUV6QixLQUFLLENBQUMyQixXQUFOLElBQXFCLEVBSHBDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFERjtBQU9ELEdBUkQsTUFRTyxJQUFJaEIsU0FBSixFQUFlO0FBQ3BCLFdBQ0UsTUFBQyw2REFBRDtBQUNFLGVBQVMsRUFBRUMsU0FEYjtBQUVFLHFCQUFlLEVBQUVGLGVBRm5CO0FBR0UsYUFBTyxFQUFFSSxtQkFIWDtBQUlFLFlBQU0sRUFBRUQsTUFKVjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BREY7QUFRRCxHQVRNLENBVVA7QUFDQTtBQUNBO0FBWk8sT0FhRixJQUNIZSwrREFBZ0IsQ0FBQ0MsWUFBakIsQ0FBOEJDLFdBQTlCLElBQ0EsQ0FBQzlCLEtBQUssQ0FBQ3lCLFlBRFAsSUFFQSxDQUFDekIsS0FBSyxDQUFDd0IsVUFISixFQUlIO0FBQ0EsYUFBTyxNQUFDLDhFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsUUFBUDtBQUNEOztBQUNELFNBQ0UsTUFBQywyRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxvRUFBRDtBQUFhLFlBQVEsTUFBckI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsa0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLHVCQURGLENBREY7QUFRRCxDQXRFTTs7R0FBTTNCLGdCO1VBQ0lFLHFELEVBRVFFLDZFLEVBS21DTSwwRDs7O0tBUi9DVixnQiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC40ZWM0ZDIxM2M1ZjYyOGQ4MWYxNy5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xuXG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vaW50ZXJmYWNlcyc7XG5pbXBvcnQgRm9sZGVyc0FuZFBsb3RzIGZyb20gJy4vZm9sZGVyc19hbmRfcGxvdHNfY29udGVudCc7XG5pbXBvcnQgeyB1c2VTZWFyY2ggfSBmcm9tICcuLi8uLi8uLi9ob29rcy91c2VTZWFyY2gnO1xuaW1wb3J0IFNlYXJjaFJlc3VsdHMgZnJvbSAnLi4vLi4vc2VhcmNoL1NlYXJjaFJlc3VsdHMnO1xuaW1wb3J0IHtcbiAgTm90Rm91bmREaXZXcmFwcGVyLFxuICBDaGFydEljb24sXG4gIE5vdEZvdW5kRGl2LFxufSBmcm9tICcuLi8uLi9zZWFyY2gvc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyBzZXBlcmF0ZVJ1bkFuZEx1bWlJblNlYXJjaCB9IGZyb20gJy4uLy4uLy4uL2NvbXBvbmVudHMvdXRpbHMnO1xuaW1wb3J0IHsgY2hhbmdlUm91dGVyLCBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMgfSBmcm9tICcuLi91dGlscyc7XG5pbXBvcnQgeyB3b3Jrc3BhY2VzIH0gZnJvbSAnLi4vLi4vLi4vd29ya3NwYWNlcy9vZmZsaW5lJztcbmltcG9ydCB7IGZ1bmN0aW9uc19jb25maWcgfSBmcm9tICcuLi8uLi8uLi9jb25maWcvY29uZmlnJztcbmltcG9ydCB7IExhdGVzdFJ1bnMgfSBmcm9tICcuLi8uLi8uLi9jb21wb25lbnRzL2luaXRpYWxQYWdlL2xhdGVzdFJ1bnMnO1xuaW1wb3J0IHsgdXNlVXBkYXRlTGl2ZU1vZGUgfSBmcm9tICcuLi8uLi8uLi9ob29rcy91c2VVcGRhdGVJbkxpdmVNb2RlJztcblxuZXhwb3J0IGNvbnN0IENvbnRlbnRTd2l0Y2hpbmcgPSAoKSA9PiB7XG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcbiAgY29uc3QgeyBzZXRfdXBkYXRlIH0gPSB1c2VVcGRhdGVMaXZlTW9kZSgpO1xuICBjb25zdCB3b3Jrc3BhY2VPcHRpb24gPSBxdWVyeS53b3Jrc3BhY2VcbiAgICA/IHF1ZXJ5LndvcmtzcGFjZVxuICAgIDogd29ya3NwYWNlc1swXS53b3Jrc3BhY2VzWzJdLmxhYmVsO1xuXG4gIGNvbnN0IHsgcmVzdWx0c19ncm91cGVkLCBzZWFyY2hpbmcsIGlzTG9hZGluZywgZXJyb3JzIH0gPSB1c2VTZWFyY2goXG4gICAgcXVlcnkuc2VhcmNoX3J1bl9udW1iZXIsXG4gICAgcXVlcnkuc2VhcmNoX2RhdGFzZXRfbmFtZVxuICApO1xuICAvL3NlcmNoUmVzdWx0c0hhbmRsZXIgd2hlbiB5b3Ugc2VsZWN0aW5nIHJ1biwgZGF0YXNldCBmcm9tIHNlYXJjaCByZXN1bHRzXG4gIGNvbnN0IHNlcmNoUmVzdWx0c0hhbmRsZXIgPSAocnVuOiBzdHJpbmcsIGRhdGFzZXQ6IHN0cmluZykgPT4ge1xuICAgIHNldF91cGRhdGUoZmFsc2UpO1xuXG4gICAgY29uc3QgeyBwYXJzZWRSdW4sIHBhcnNlZEx1bWkgfSA9IHNlcGVyYXRlUnVuQW5kTHVtaUluU2VhcmNoKFxuICAgICAgcnVuLnRvU3RyaW5nKClcbiAgICApO1xuICAgIFxuICAgIGNoYW5nZVJvdXRlcihcbiAgICAgIGdldENoYW5nZWRRdWVyeVBhcmFtcyhcbiAgICAgICAge1xuICAgICAgICAgIGx1bWk6IHBhcnNlZEx1bWksXG4gICAgICAgICAgcnVuX251bWJlcjogcGFyc2VkUnVuLFxuICAgICAgICAgIGRhdGFzZXRfbmFtZTogZGF0YXNldCxcbiAgICAgICAgICB3b3Jrc3BhY2VzOiB3b3Jrc3BhY2VPcHRpb24sXG4gICAgICAgICAgcGxvdF9zZWFyY2g6ICcnLFxuICAgICAgICB9LFxuICAgICAgICBxdWVyeVxuICAgICAgKVxuICAgICk7XG4gIH07XG5cbiAgaWYgKHF1ZXJ5LmRhdGFzZXRfbmFtZSAmJiBxdWVyeS5ydW5fbnVtYmVyKSB7XG4gICAgcmV0dXJuIChcbiAgICAgIDxGb2xkZXJzQW5kUGxvdHNcbiAgICAgICAgcnVuX251bWJlcj17cXVlcnkucnVuX251bWJlciB8fCAnJ31cbiAgICAgICAgZGF0YXNldF9uYW1lPXtxdWVyeS5kYXRhc2V0X25hbWUgfHwgJyd9XG4gICAgICAgIGZvbGRlcl9wYXRoPXtxdWVyeS5mb2xkZXJfcGF0aCB8fCAnJ31cbiAgICAgIC8+XG4gICAgKTtcbiAgfSBlbHNlIGlmIChzZWFyY2hpbmcpIHtcbiAgICByZXR1cm4gKFxuICAgICAgPFNlYXJjaFJlc3VsdHNcbiAgICAgICAgaXNMb2FkaW5nPXtpc0xvYWRpbmd9XG4gICAgICAgIHJlc3VsdHNfZ3JvdXBlZD17cmVzdWx0c19ncm91cGVkfVxuICAgICAgICBoYW5kbGVyPXtzZXJjaFJlc3VsdHNIYW5kbGVyfVxuICAgICAgICBlcnJvcnM9e2Vycm9yc31cbiAgICAgIC8+XG4gICAgKTtcbiAgfVxuICAvLyAhcXVlcnkuZGF0YXNldF9uYW1lICYmICFxdWVyeS5ydW5fbnVtYmVyIGJlY2F1c2UgSSBkb24ndCB3YW50XG4gIC8vIHRvIHNlZSBsYXRlc3QgcnVucyBsaXN0LCB3aGVuIEknbSBsb2FkaW5nIGZvbGRlcnMgb3IgcGxvdHNcbiAgLy8gIGZvbGRlcnMgYW5kICBwbG90cyBhcmUgdmlzaWJsZSwgd2hlbiBkYXRhc2V0X25hbWUgYW5kIHJ1bl9udW1iZXIgaXMgc2V0XG4gIGVsc2UgaWYgKFxuICAgIGZ1bmN0aW9uc19jb25maWcubmV3X2JhY2tfZW5kLmxhdGVzdF9ydW5zICYmXG4gICAgIXF1ZXJ5LmRhdGFzZXRfbmFtZSAmJlxuICAgICFxdWVyeS5ydW5fbnVtYmVyXG4gICkge1xuICAgIHJldHVybiA8TGF0ZXN0UnVucyAvPjtcbiAgfVxuICByZXR1cm4gKFxuICAgIDxOb3RGb3VuZERpdldyYXBwZXI+XG4gICAgICA8Tm90Rm91bmREaXYgbm9Cb3JkZXI+XG4gICAgICAgIDxDaGFydEljb24gLz5cbiAgICAgICAgV2VsY29tZSB0byBEUU0gR1VJXG4gICAgICA8L05vdEZvdW5kRGl2PlxuICAgIDwvTm90Rm91bmREaXZXcmFwcGVyPlxuICApO1xufTtcbiJdLCJzb3VyY2VSb290IjoiIn0=