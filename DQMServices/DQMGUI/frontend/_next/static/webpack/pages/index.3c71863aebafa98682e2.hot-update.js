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
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../../../config/config */ "./config/config.ts");
/* harmony import */ var _components_initialPage_latestRuns__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../../../components/initialPage/latestRuns */ "./components/initialPage/latestRuns.tsx");
/* harmony import */ var _hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../../../hooks/useUpdateInLiveMode */ "./hooks/useUpdateInLiveMode.tsx");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ../../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/containers/display/content/constent_switching.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];












var ContentSwitching = function ContentSwitching() {
  _s();

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_1__["useRouter"])();
  var query = router.query;

  var _useUpdateLiveMode = Object(_hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_10__["useUpdateLiveMode"])(),
      set_update = _useUpdateLiveMode.set_update;

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_0__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_11__["store"]),
      wokrspace = _React$useContext.wokrspace;

  var _useSearch = Object(_hooks_useSearch__WEBPACK_IMPORTED_MODULE_3__["useSearch"])(query.search_run_number, query.search_dataset_name, query.search_by_lumisection),
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
      workspaces: wokrspace,
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
  else if (_config_config__WEBPACK_IMPORTED_MODULE_8__["functions_config"].new_back_end.latest_runs && !query.dataset_name && !query.run_number) {
      return __jsx(_components_initialPage_latestRuns__WEBPACK_IMPORTED_MODULE_9__["LatestRuns"], {
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

_s(ContentSwitching, "wLpK/YwrHs3aa3rwx2mALqPX4vw=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_1__["useRouter"], _hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_10__["useUpdateLiveMode"], _hooks_useSearch__WEBPACK_IMPORTED_MODULE_3__["useSearch"]];
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGFpbmVycy9kaXNwbGF5L2NvbnRlbnQvY29uc3RlbnRfc3dpdGNoaW5nLnRzeCJdLCJuYW1lcyI6WyJDb250ZW50U3dpdGNoaW5nIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJ1c2VVcGRhdGVMaXZlTW9kZSIsInNldF91cGRhdGUiLCJSZWFjdCIsInN0b3JlIiwid29rcnNwYWNlIiwidXNlU2VhcmNoIiwic2VhcmNoX3J1bl9udW1iZXIiLCJzZWFyY2hfZGF0YXNldF9uYW1lIiwic2VhcmNoX2J5X2x1bWlzZWN0aW9uIiwicmVzdWx0c19ncm91cGVkIiwic2VhcmNoaW5nIiwiaXNMb2FkaW5nIiwiZXJyb3JzIiwic2VyY2hSZXN1bHRzSGFuZGxlciIsInJ1biIsImRhdGFzZXQiLCJzZXBlcmF0ZVJ1bkFuZEx1bWlJblNlYXJjaCIsInRvU3RyaW5nIiwicGFyc2VkUnVuIiwicGFyc2VkTHVtaSIsImNoYW5nZVJvdXRlciIsImdldENoYW5nZWRRdWVyeVBhcmFtcyIsImx1bWkiLCJydW5fbnVtYmVyIiwiZGF0YXNldF9uYW1lIiwid29ya3NwYWNlcyIsInBsb3Rfc2VhcmNoIiwiZm9sZGVyX3BhdGgiLCJmdW5jdGlvbnNfY29uZmlnIiwibmV3X2JhY2tfZW5kIiwibGF0ZXN0X3J1bnMiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBR0E7QUFDQTtBQUNBO0FBQ0E7QUFLQTtBQUNBO0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFFTyxJQUFNQSxnQkFBZ0IsR0FBRyxTQUFuQkEsZ0JBQW1CLEdBQU07QUFBQTs7QUFDcEMsTUFBTUMsTUFBTSxHQUFHQyw2REFBUyxFQUF4QjtBQUNBLE1BQU1DLEtBQWlCLEdBQUdGLE1BQU0sQ0FBQ0UsS0FBakM7O0FBRm9DLDJCQUdiQyxxRkFBaUIsRUFISjtBQUFBLE1BRzVCQyxVQUg0QixzQkFHNUJBLFVBSDRCOztBQUFBLDBCQUlkQyxnREFBQSxDQUFpQkMsZ0VBQWpCLENBSmM7QUFBQSxNQUk1QkMsU0FKNEIscUJBSTVCQSxTQUo0Qjs7QUFBQSxtQkFNc0JDLGtFQUFTLENBQ2pFTixLQUFLLENBQUNPLGlCQUQyRCxFQUVqRVAsS0FBSyxDQUFDUSxtQkFGMkQsRUFHakVSLEtBQUssQ0FBQ1MscUJBSDJELENBTi9CO0FBQUEsTUFNNUJDLGVBTjRCLGNBTTVCQSxlQU40QjtBQUFBLE1BTVhDLFNBTlcsY0FNWEEsU0FOVztBQUFBLE1BTUFDLFNBTkEsY0FNQUEsU0FOQTtBQUFBLE1BTVdDLE1BTlgsY0FNV0EsTUFOWCxFQVdwQzs7O0FBQ0EsTUFBTUMsbUJBQW1CLEdBQUcsU0FBdEJBLG1CQUFzQixDQUFDQyxHQUFELEVBQWNDLE9BQWQsRUFBa0M7QUFDNURkLGNBQVUsQ0FBQyxLQUFELENBQVY7O0FBRDRELGdDQUcxQmUsb0ZBQTBCLENBQzFERixHQUFHLENBQUNHLFFBQUosRUFEMEQsQ0FIQTtBQUFBLFFBR3BEQyxTQUhvRCx5QkFHcERBLFNBSG9EO0FBQUEsUUFHekNDLFVBSHlDLHlCQUd6Q0EsVUFIeUM7O0FBTzVEQywrREFBWSxDQUNWQyxvRUFBcUIsQ0FDbkI7QUFDRUMsVUFBSSxFQUFFSCxVQURSO0FBRUVJLGdCQUFVLEVBQUVMLFNBRmQ7QUFHRU0sa0JBQVksRUFBRVQsT0FIaEI7QUFJRVUsZ0JBQVUsRUFBRXJCLFNBSmQ7QUFLRXNCLGlCQUFXLEVBQUU7QUFMZixLQURtQixFQVFuQjNCLEtBUm1CLENBRFgsQ0FBWjtBQVlELEdBbkJEOztBQXFCQSxNQUFJQSxLQUFLLENBQUN5QixZQUFOLElBQXNCekIsS0FBSyxDQUFDd0IsVUFBaEMsRUFBNEM7QUFDMUMsV0FDRSxNQUFDLGtFQUFEO0FBQ0UsZ0JBQVUsRUFBRXhCLEtBQUssQ0FBQ3dCLFVBQU4sSUFBb0IsRUFEbEM7QUFFRSxrQkFBWSxFQUFFeEIsS0FBSyxDQUFDeUIsWUFBTixJQUFzQixFQUZ0QztBQUdFLGlCQUFXLEVBQUV6QixLQUFLLENBQUM0QixXQUFOLElBQXFCLEVBSHBDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFERjtBQU9ELEdBUkQsTUFRTyxJQUFJakIsU0FBSixFQUFlO0FBQ3BCLFdBQ0UsTUFBQyw2REFBRDtBQUNFLGVBQVMsRUFBRUMsU0FEYjtBQUVFLHFCQUFlLEVBQUVGLGVBRm5CO0FBR0UsYUFBTyxFQUFFSSxtQkFIWDtBQUlFLFlBQU0sRUFBRUQsTUFKVjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BREY7QUFRRCxHQVRNLENBVVA7QUFDQTtBQUNBO0FBWk8sT0FhRixJQUNIZ0IsK0RBQWdCLENBQUNDLFlBQWpCLENBQThCQyxXQUE5QixJQUNBLENBQUMvQixLQUFLLENBQUN5QixZQURQLElBRUEsQ0FBQ3pCLEtBQUssQ0FBQ3dCLFVBSEosRUFJSDtBQUNBLGFBQU8sTUFBQyw2RUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFFBQVA7QUFDRDs7QUFDRCxTQUNFLE1BQUMsMkVBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsb0VBQUQ7QUFBYSxZQUFRLE1BQXJCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLGtFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERix1QkFERixDQURGO0FBUUQsQ0FyRU07O0dBQU0zQixnQjtVQUNJRSxxRCxFQUVRRSw2RSxFQUdtQ0ssMEQ7OztLQU4vQ1QsZ0IiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguM2M3MTg2M2FlYmFmYTk4NjgyZTIuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IHVzZVJvdXRlciB9IGZyb20gJ25leHQvcm91dGVyJztcblxuaW1wb3J0IHsgUXVlcnlQcm9wcyB9IGZyb20gJy4uL2ludGVyZmFjZXMnO1xuaW1wb3J0IEZvbGRlcnNBbmRQbG90cyBmcm9tICcuL2ZvbGRlcnNfYW5kX3Bsb3RzX2NvbnRlbnQnO1xuaW1wb3J0IHsgdXNlU2VhcmNoIH0gZnJvbSAnLi4vLi4vLi4vaG9va3MvdXNlU2VhcmNoJztcbmltcG9ydCBTZWFyY2hSZXN1bHRzIGZyb20gJy4uLy4uL3NlYXJjaC9TZWFyY2hSZXN1bHRzJztcbmltcG9ydCB7XG4gIE5vdEZvdW5kRGl2V3JhcHBlcixcbiAgQ2hhcnRJY29uLFxuICBOb3RGb3VuZERpdixcbn0gZnJvbSAnLi4vLi4vc2VhcmNoL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgc2VwZXJhdGVSdW5BbmRMdW1pSW5TZWFyY2ggfSBmcm9tICcuLi8uLi8uLi9jb21wb25lbnRzL3V0aWxzJztcbmltcG9ydCB7IGNoYW5nZVJvdXRlciwgZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zIH0gZnJvbSAnLi4vdXRpbHMnO1xuaW1wb3J0IHsgd29ya3NwYWNlcyB9IGZyb20gJy4uLy4uLy4uL3dvcmtzcGFjZXMvb2ZmbGluZSc7XG5pbXBvcnQgeyBmdW5jdGlvbnNfY29uZmlnIH0gZnJvbSAnLi4vLi4vLi4vY29uZmlnL2NvbmZpZyc7XG5pbXBvcnQgeyBMYXRlc3RSdW5zIH0gZnJvbSAnLi4vLi4vLi4vY29tcG9uZW50cy9pbml0aWFsUGFnZS9sYXRlc3RSdW5zJztcbmltcG9ydCB7IHVzZVVwZGF0ZUxpdmVNb2RlIH0gZnJvbSAnLi4vLi4vLi4vaG9va3MvdXNlVXBkYXRlSW5MaXZlTW9kZSc7XG5pbXBvcnQgeyBzdG9yZSB9IGZyb20gJy4uLy4uLy4uL2NvbnRleHRzL2xlZnRTaWRlQ29udGV4dCc7XG5cbmV4cG9ydCBjb25zdCBDb250ZW50U3dpdGNoaW5nID0gKCkgPT4ge1xuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XG4gIGNvbnN0IHsgc2V0X3VwZGF0ZSB9ID0gdXNlVXBkYXRlTGl2ZU1vZGUoKTtcbiAgY29uc3QgeyB3b2tyc3BhY2UgfSA9IFJlYWN0LnVzZUNvbnRleHQoc3RvcmUpXG5cbiAgY29uc3QgeyByZXN1bHRzX2dyb3VwZWQsIHNlYXJjaGluZywgaXNMb2FkaW5nLCBlcnJvcnMgfSA9IHVzZVNlYXJjaChcbiAgICBxdWVyeS5zZWFyY2hfcnVuX251bWJlcixcbiAgICBxdWVyeS5zZWFyY2hfZGF0YXNldF9uYW1lLFxuICAgIHF1ZXJ5LnNlYXJjaF9ieV9sdW1pc2VjdGlvbixcbiAgKTtcbiAgLy9zZXJjaFJlc3VsdHNIYW5kbGVyIHdoZW4geW91IHNlbGVjdGluZyBydW4sIGRhdGFzZXQgZnJvbSBzZWFyY2ggcmVzdWx0c1xuICBjb25zdCBzZXJjaFJlc3VsdHNIYW5kbGVyID0gKHJ1bjogc3RyaW5nLCBkYXRhc2V0OiBzdHJpbmcpID0+IHtcbiAgICBzZXRfdXBkYXRlKGZhbHNlKTtcblxuICAgIGNvbnN0IHsgcGFyc2VkUnVuLCBwYXJzZWRMdW1pIH0gPSBzZXBlcmF0ZVJ1bkFuZEx1bWlJblNlYXJjaChcbiAgICAgIHJ1bi50b1N0cmluZygpXG4gICAgKTtcblxuICAgIGNoYW5nZVJvdXRlcihcbiAgICAgIGdldENoYW5nZWRRdWVyeVBhcmFtcyhcbiAgICAgICAge1xuICAgICAgICAgIGx1bWk6IHBhcnNlZEx1bWksXG4gICAgICAgICAgcnVuX251bWJlcjogcGFyc2VkUnVuLFxuICAgICAgICAgIGRhdGFzZXRfbmFtZTogZGF0YXNldCxcbiAgICAgICAgICB3b3Jrc3BhY2VzOiB3b2tyc3BhY2UsXG4gICAgICAgICAgcGxvdF9zZWFyY2g6ICcnLFxuICAgICAgICB9LFxuICAgICAgICBxdWVyeVxuICAgICAgKVxuICAgICk7XG4gIH07XG5cbiAgaWYgKHF1ZXJ5LmRhdGFzZXRfbmFtZSAmJiBxdWVyeS5ydW5fbnVtYmVyKSB7XG4gICAgcmV0dXJuIChcbiAgICAgIDxGb2xkZXJzQW5kUGxvdHNcbiAgICAgICAgcnVuX251bWJlcj17cXVlcnkucnVuX251bWJlciB8fCAnJ31cbiAgICAgICAgZGF0YXNldF9uYW1lPXtxdWVyeS5kYXRhc2V0X25hbWUgfHwgJyd9XG4gICAgICAgIGZvbGRlcl9wYXRoPXtxdWVyeS5mb2xkZXJfcGF0aCB8fCAnJ31cbiAgICAgIC8+XG4gICAgKTtcbiAgfSBlbHNlIGlmIChzZWFyY2hpbmcpIHtcbiAgICByZXR1cm4gKFxuICAgICAgPFNlYXJjaFJlc3VsdHNcbiAgICAgICAgaXNMb2FkaW5nPXtpc0xvYWRpbmd9XG4gICAgICAgIHJlc3VsdHNfZ3JvdXBlZD17cmVzdWx0c19ncm91cGVkfVxuICAgICAgICBoYW5kbGVyPXtzZXJjaFJlc3VsdHNIYW5kbGVyfVxuICAgICAgICBlcnJvcnM9e2Vycm9yc31cbiAgICAgIC8+XG4gICAgKTtcbiAgfVxuICAvLyAhcXVlcnkuZGF0YXNldF9uYW1lICYmICFxdWVyeS5ydW5fbnVtYmVyIGJlY2F1c2UgSSBkb24ndCB3YW50XG4gIC8vIHRvIHNlZSBsYXRlc3QgcnVucyBsaXN0LCB3aGVuIEknbSBsb2FkaW5nIGZvbGRlcnMgb3IgcGxvdHNcbiAgLy8gIGZvbGRlcnMgYW5kICBwbG90cyBhcmUgdmlzaWJsZSwgd2hlbiBkYXRhc2V0X25hbWUgYW5kIHJ1bl9udW1iZXIgaXMgc2V0XG4gIGVsc2UgaWYgKFxuICAgIGZ1bmN0aW9uc19jb25maWcubmV3X2JhY2tfZW5kLmxhdGVzdF9ydW5zICYmXG4gICAgIXF1ZXJ5LmRhdGFzZXRfbmFtZSAmJlxuICAgICFxdWVyeS5ydW5fbnVtYmVyXG4gICkge1xuICAgIHJldHVybiA8TGF0ZXN0UnVucyAvPjtcbiAgfVxuICByZXR1cm4gKFxuICAgIDxOb3RGb3VuZERpdldyYXBwZXI+XG4gICAgICA8Tm90Rm91bmREaXYgbm9Cb3JkZXI+XG4gICAgICAgIDxDaGFydEljb24gLz5cbiAgICAgICAgV2VsY29tZSB0byBEUU0gR1VJXG4gICAgICA8L05vdEZvdW5kRGl2PlxuICAgIDwvTm90Rm91bmREaXZXcmFwcGVyPlxuICApO1xufTtcbiJdLCJzb3VyY2VSb290IjoiIn0=