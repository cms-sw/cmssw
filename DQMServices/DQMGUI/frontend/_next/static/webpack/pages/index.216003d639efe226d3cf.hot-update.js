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
        lineNumber: 54,
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
        lineNumber: 62,
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
          lineNumber: 78,
          columnNumber: 12
        }
      });
    }

  return __jsx(_search_styledComponents__WEBPACK_IMPORTED_MODULE_5__["NotFoundDivWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 81,
      columnNumber: 5
    }
  }, __jsx(_search_styledComponents__WEBPACK_IMPORTED_MODULE_5__["NotFoundDiv"], {
    noBorder: true,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 82,
      columnNumber: 7
    }
  }, __jsx(_search_styledComponents__WEBPACK_IMPORTED_MODULE_5__["ChartIcon"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 83,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGFpbmVycy9kaXNwbGF5L2NvbnRlbnQvY29uc3RlbnRfc3dpdGNoaW5nLnRzeCJdLCJuYW1lcyI6WyJDb250ZW50U3dpdGNoaW5nIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJ1c2VVcGRhdGVMaXZlTW9kZSIsInNldF91cGRhdGUiLCJSZWFjdCIsInN0b3JlIiwid29rcnNwYWNlIiwidXNlU2VhcmNoIiwic2VhcmNoX3J1bl9udW1iZXIiLCJzZWFyY2hfZGF0YXNldF9uYW1lIiwicmVzdWx0c19ncm91cGVkIiwic2VhcmNoaW5nIiwiaXNMb2FkaW5nIiwiZXJyb3JzIiwic2VyY2hSZXN1bHRzSGFuZGxlciIsInJ1biIsImRhdGFzZXQiLCJzZXBlcmF0ZVJ1bkFuZEx1bWlJblNlYXJjaCIsInRvU3RyaW5nIiwicGFyc2VkUnVuIiwicGFyc2VkTHVtaSIsImNoYW5nZVJvdXRlciIsImdldENoYW5nZWRRdWVyeVBhcmFtcyIsImx1bWkiLCJydW5fbnVtYmVyIiwiZGF0YXNldF9uYW1lIiwid29ya3NwYWNlcyIsInBsb3Rfc2VhcmNoIiwiZm9sZGVyX3BhdGgiLCJmdW5jdGlvbnNfY29uZmlnIiwibmV3X2JhY2tfZW5kIiwibGF0ZXN0X3J1bnMiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBR0E7QUFDQTtBQUNBO0FBQ0E7QUFLQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFFTyxJQUFNQSxnQkFBZ0IsR0FBRyxTQUFuQkEsZ0JBQW1CLEdBQU07QUFBQTs7QUFDcEMsTUFBTUMsTUFBTSxHQUFHQyw2REFBUyxFQUF4QjtBQUNBLE1BQU1DLEtBQWlCLEdBQUdGLE1BQU0sQ0FBQ0UsS0FBakM7O0FBRm9DLDJCQUdiQyxxRkFBaUIsRUFISjtBQUFBLE1BRzVCQyxVQUg0QixzQkFHNUJBLFVBSDRCOztBQUFBLDBCQUlkQyxnREFBQSxDQUFpQkMsZ0VBQWpCLENBSmM7QUFBQSxNQUk1QkMsU0FKNEIscUJBSTVCQSxTQUo0Qjs7QUFBQSxtQkFNc0JDLGtFQUFTLENBQ2pFTixLQUFLLENBQUNPLGlCQUQyRCxFQUVqRVAsS0FBSyxDQUFDUSxtQkFGMkQsQ0FOL0I7QUFBQSxNQU01QkMsZUFONEIsY0FNNUJBLGVBTjRCO0FBQUEsTUFNWEMsU0FOVyxjQU1YQSxTQU5XO0FBQUEsTUFNQUMsU0FOQSxjQU1BQSxTQU5BO0FBQUEsTUFNV0MsTUFOWCxjQU1XQSxNQU5YLEVBVXBDOzs7QUFDQSxNQUFNQyxtQkFBbUIsR0FBRyxTQUF0QkEsbUJBQXNCLENBQUNDLEdBQUQsRUFBY0MsT0FBZCxFQUFrQztBQUM1RGIsY0FBVSxDQUFDLEtBQUQsQ0FBVjs7QUFENEQsZ0NBRzFCYyxvRkFBMEIsQ0FDMURGLEdBQUcsQ0FBQ0csUUFBSixFQUQwRCxDQUhBO0FBQUEsUUFHcERDLFNBSG9ELHlCQUdwREEsU0FIb0Q7QUFBQSxRQUd6Q0MsVUFIeUMseUJBR3pDQSxVQUh5Qzs7QUFPNURDLCtEQUFZLENBQ1ZDLG9FQUFxQixDQUNuQjtBQUNFQyxVQUFJLEVBQUVILFVBRFI7QUFFRUksZ0JBQVUsRUFBRUwsU0FGZDtBQUdFTSxrQkFBWSxFQUFFVCxPQUhoQjtBQUlFVSxnQkFBVSxFQUFFcEIsU0FKZDtBQUtFcUIsaUJBQVcsRUFBRTtBQUxmLEtBRG1CLEVBUW5CMUIsS0FSbUIsQ0FEWCxDQUFaO0FBWUQsR0FuQkQ7O0FBcUJBLE1BQUlBLEtBQUssQ0FBQ3dCLFlBQU4sSUFBc0J4QixLQUFLLENBQUN1QixVQUFoQyxFQUE0QztBQUMxQyxXQUNFLE1BQUMsa0VBQUQ7QUFDRSxnQkFBVSxFQUFFdkIsS0FBSyxDQUFDdUIsVUFBTixJQUFvQixFQURsQztBQUVFLGtCQUFZLEVBQUV2QixLQUFLLENBQUN3QixZQUFOLElBQXNCLEVBRnRDO0FBR0UsaUJBQVcsRUFBRXhCLEtBQUssQ0FBQzJCLFdBQU4sSUFBcUIsRUFIcEM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQURGO0FBT0QsR0FSRCxNQVFPLElBQUlqQixTQUFKLEVBQWU7QUFDcEIsV0FDRSxNQUFDLDZEQUFEO0FBQ0UsZUFBUyxFQUFFQyxTQURiO0FBRUUscUJBQWUsRUFBRUYsZUFGbkI7QUFHRSxhQUFPLEVBQUVJLG1CQUhYO0FBSUUsWUFBTSxFQUFFRCxNQUpWO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFERjtBQVFELEdBVE0sQ0FVUDtBQUNBO0FBQ0E7QUFaTyxPQWFGLElBQ0hnQiwrREFBZ0IsQ0FBQ0MsWUFBakIsQ0FBOEJDLFdBQTlCLElBQ0EsQ0FBQzlCLEtBQUssQ0FBQ3dCLFlBRFAsSUFFQSxDQUFDeEIsS0FBSyxDQUFDdUIsVUFISixFQUlIO0FBQ0EsYUFBTyxNQUFDLDZFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsUUFBUDtBQUNEOztBQUNELFNBQ0UsTUFBQywyRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxvRUFBRDtBQUFhLFlBQVEsTUFBckI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsa0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLHVCQURGLENBREY7QUFRRCxDQXBFTTs7R0FBTTFCLGdCO1VBQ0lFLHFELEVBRVFFLDZFLEVBR21DSywwRDs7O0tBTi9DVCxnQiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4yMTYwMDNkNjM5ZWZlMjI2ZDNjZi5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xuXG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vaW50ZXJmYWNlcyc7XG5pbXBvcnQgRm9sZGVyc0FuZFBsb3RzIGZyb20gJy4vZm9sZGVyc19hbmRfcGxvdHNfY29udGVudCc7XG5pbXBvcnQgeyB1c2VTZWFyY2ggfSBmcm9tICcuLi8uLi8uLi9ob29rcy91c2VTZWFyY2gnO1xuaW1wb3J0IFNlYXJjaFJlc3VsdHMgZnJvbSAnLi4vLi4vc2VhcmNoL1NlYXJjaFJlc3VsdHMnO1xuaW1wb3J0IHtcbiAgTm90Rm91bmREaXZXcmFwcGVyLFxuICBDaGFydEljb24sXG4gIE5vdEZvdW5kRGl2LFxufSBmcm9tICcuLi8uLi9zZWFyY2gvc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyBzZXBlcmF0ZVJ1bkFuZEx1bWlJblNlYXJjaCB9IGZyb20gJy4uLy4uLy4uL2NvbXBvbmVudHMvdXRpbHMnO1xuaW1wb3J0IHsgY2hhbmdlUm91dGVyLCBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMgfSBmcm9tICcuLi91dGlscyc7XG5pbXBvcnQgeyBmdW5jdGlvbnNfY29uZmlnIH0gZnJvbSAnLi4vLi4vLi4vY29uZmlnL2NvbmZpZyc7XG5pbXBvcnQgeyBMYXRlc3RSdW5zIH0gZnJvbSAnLi4vLi4vLi4vY29tcG9uZW50cy9pbml0aWFsUGFnZS9sYXRlc3RSdW5zJztcbmltcG9ydCB7IHVzZVVwZGF0ZUxpdmVNb2RlIH0gZnJvbSAnLi4vLi4vLi4vaG9va3MvdXNlVXBkYXRlSW5MaXZlTW9kZSc7XG5pbXBvcnQgeyBzdG9yZSB9IGZyb20gJy4uLy4uLy4uL2NvbnRleHRzL2xlZnRTaWRlQ29udGV4dCc7XG5cbmV4cG9ydCBjb25zdCBDb250ZW50U3dpdGNoaW5nID0gKCkgPT4ge1xuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XG4gIGNvbnN0IHsgc2V0X3VwZGF0ZSB9ID0gdXNlVXBkYXRlTGl2ZU1vZGUoKTtcbiAgY29uc3QgeyB3b2tyc3BhY2UgfSA9IFJlYWN0LnVzZUNvbnRleHQoc3RvcmUpXG5cbiAgY29uc3QgeyByZXN1bHRzX2dyb3VwZWQsIHNlYXJjaGluZywgaXNMb2FkaW5nLCBlcnJvcnMgfSA9IHVzZVNlYXJjaChcbiAgICBxdWVyeS5zZWFyY2hfcnVuX251bWJlcixcbiAgICBxdWVyeS5zZWFyY2hfZGF0YXNldF9uYW1lLFxuICApO1xuICAvL3NlcmNoUmVzdWx0c0hhbmRsZXIgd2hlbiB5b3Ugc2VsZWN0aW5nIHJ1biwgZGF0YXNldCBmcm9tIHNlYXJjaCByZXN1bHRzXG4gIGNvbnN0IHNlcmNoUmVzdWx0c0hhbmRsZXIgPSAocnVuOiBzdHJpbmcsIGRhdGFzZXQ6IHN0cmluZykgPT4ge1xuICAgIHNldF91cGRhdGUoZmFsc2UpO1xuXG4gICAgY29uc3QgeyBwYXJzZWRSdW4sIHBhcnNlZEx1bWkgfSA9IHNlcGVyYXRlUnVuQW5kTHVtaUluU2VhcmNoKFxuICAgICAgcnVuLnRvU3RyaW5nKClcbiAgICApO1xuXG4gICAgY2hhbmdlUm91dGVyKFxuICAgICAgZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zKFxuICAgICAgICB7XG4gICAgICAgICAgbHVtaTogcGFyc2VkTHVtaSxcbiAgICAgICAgICBydW5fbnVtYmVyOiBwYXJzZWRSdW4sXG4gICAgICAgICAgZGF0YXNldF9uYW1lOiBkYXRhc2V0LFxuICAgICAgICAgIHdvcmtzcGFjZXM6IHdva3JzcGFjZSxcbiAgICAgICAgICBwbG90X3NlYXJjaDogJycsXG4gICAgICAgIH0sXG4gICAgICAgIHF1ZXJ5XG4gICAgICApXG4gICAgKTtcbiAgfTtcblxuICBpZiAocXVlcnkuZGF0YXNldF9uYW1lICYmIHF1ZXJ5LnJ1bl9udW1iZXIpIHtcbiAgICByZXR1cm4gKFxuICAgICAgPEZvbGRlcnNBbmRQbG90c1xuICAgICAgICBydW5fbnVtYmVyPXtxdWVyeS5ydW5fbnVtYmVyIHx8ICcnfVxuICAgICAgICBkYXRhc2V0X25hbWU9e3F1ZXJ5LmRhdGFzZXRfbmFtZSB8fCAnJ31cbiAgICAgICAgZm9sZGVyX3BhdGg9e3F1ZXJ5LmZvbGRlcl9wYXRoIHx8ICcnfVxuICAgICAgLz5cbiAgICApO1xuICB9IGVsc2UgaWYgKHNlYXJjaGluZykge1xuICAgIHJldHVybiAoXG4gICAgICA8U2VhcmNoUmVzdWx0c1xuICAgICAgICBpc0xvYWRpbmc9e2lzTG9hZGluZ31cbiAgICAgICAgcmVzdWx0c19ncm91cGVkPXtyZXN1bHRzX2dyb3VwZWR9XG4gICAgICAgIGhhbmRsZXI9e3NlcmNoUmVzdWx0c0hhbmRsZXJ9XG4gICAgICAgIGVycm9ycz17ZXJyb3JzfVxuICAgICAgLz5cbiAgICApO1xuICB9XG4gIC8vICFxdWVyeS5kYXRhc2V0X25hbWUgJiYgIXF1ZXJ5LnJ1bl9udW1iZXIgYmVjYXVzZSBJIGRvbid0IHdhbnRcbiAgLy8gdG8gc2VlIGxhdGVzdCBydW5zIGxpc3QsIHdoZW4gSSdtIGxvYWRpbmcgZm9sZGVycyBvciBwbG90c1xuICAvLyAgZm9sZGVycyBhbmQgIHBsb3RzIGFyZSB2aXNpYmxlLCB3aGVuIGRhdGFzZXRfbmFtZSBhbmQgcnVuX251bWJlciBpcyBzZXRcbiAgZWxzZSBpZiAoXG4gICAgZnVuY3Rpb25zX2NvbmZpZy5uZXdfYmFja19lbmQubGF0ZXN0X3J1bnMgJiZcbiAgICAhcXVlcnkuZGF0YXNldF9uYW1lICYmXG4gICAgIXF1ZXJ5LnJ1bl9udW1iZXJcbiAgKSB7XG4gICAgcmV0dXJuIDxMYXRlc3RSdW5zIC8+O1xuICB9XG4gIHJldHVybiAoXG4gICAgPE5vdEZvdW5kRGl2V3JhcHBlcj5cbiAgICAgIDxOb3RGb3VuZERpdiBub0JvcmRlcj5cbiAgICAgICAgPENoYXJ0SWNvbiAvPlxuICAgICAgICBXZWxjb21lIHRvIERRTSBHVUlcbiAgICAgIDwvTm90Rm91bmREaXY+XG4gICAgPC9Ob3RGb3VuZERpdldyYXBwZXI+XG4gICk7XG59O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==