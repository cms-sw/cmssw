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
        lineNumber: 55,
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
        lineNumber: 63,
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
          lineNumber: 79,
          columnNumber: 12
        }
      });
    }

  return __jsx(_search_styledComponents__WEBPACK_IMPORTED_MODULE_5__["NotFoundDivWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 82,
      columnNumber: 5
    }
  }, __jsx(_search_styledComponents__WEBPACK_IMPORTED_MODULE_5__["NotFoundDiv"], {
    noBorder: true,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 83,
      columnNumber: 7
    }
  }, __jsx(_search_styledComponents__WEBPACK_IMPORTED_MODULE_5__["ChartIcon"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 84,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGFpbmVycy9kaXNwbGF5L2NvbnRlbnQvY29uc3RlbnRfc3dpdGNoaW5nLnRzeCJdLCJuYW1lcyI6WyJDb250ZW50U3dpdGNoaW5nIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJ1c2VVcGRhdGVMaXZlTW9kZSIsInNldF91cGRhdGUiLCJSZWFjdCIsInN0b3JlIiwid29rcnNwYWNlIiwidXNlU2VhcmNoIiwic2VhcmNoX3J1bl9udW1iZXIiLCJzZWFyY2hfZGF0YXNldF9uYW1lIiwicmVzdWx0c19ncm91cGVkIiwic2VhcmNoaW5nIiwiaXNMb2FkaW5nIiwiZXJyb3JzIiwic2VyY2hSZXN1bHRzSGFuZGxlciIsInJ1biIsImRhdGFzZXQiLCJzZXBlcmF0ZVJ1bkFuZEx1bWlJblNlYXJjaCIsInRvU3RyaW5nIiwicGFyc2VkUnVuIiwicGFyc2VkTHVtaSIsImNoYW5nZVJvdXRlciIsImdldENoYW5nZWRRdWVyeVBhcmFtcyIsImx1bWkiLCJydW5fbnVtYmVyIiwiZGF0YXNldF9uYW1lIiwid29ya3NwYWNlcyIsInBsb3Rfc2VhcmNoIiwiZm9sZGVyX3BhdGgiLCJmdW5jdGlvbnNfY29uZmlnIiwibmV3X2JhY2tfZW5kIiwibGF0ZXN0X3J1bnMiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBR0E7QUFDQTtBQUNBO0FBQ0E7QUFLQTtBQUNBO0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFFTyxJQUFNQSxnQkFBZ0IsR0FBRyxTQUFuQkEsZ0JBQW1CLEdBQU07QUFBQTs7QUFDcEMsTUFBTUMsTUFBTSxHQUFHQyw2REFBUyxFQUF4QjtBQUNBLE1BQU1DLEtBQWlCLEdBQUdGLE1BQU0sQ0FBQ0UsS0FBakM7O0FBRm9DLDJCQUdiQyxxRkFBaUIsRUFISjtBQUFBLE1BRzVCQyxVQUg0QixzQkFHNUJBLFVBSDRCOztBQUFBLDBCQUlkQyxnREFBQSxDQUFpQkMsZ0VBQWpCLENBSmM7QUFBQSxNQUk1QkMsU0FKNEIscUJBSTVCQSxTQUo0Qjs7QUFBQSxtQkFNc0JDLGtFQUFTLENBQ2pFTixLQUFLLENBQUNPLGlCQUQyRCxFQUVqRVAsS0FBSyxDQUFDUSxtQkFGMkQsQ0FOL0I7QUFBQSxNQU01QkMsZUFONEIsY0FNNUJBLGVBTjRCO0FBQUEsTUFNWEMsU0FOVyxjQU1YQSxTQU5XO0FBQUEsTUFNQUMsU0FOQSxjQU1BQSxTQU5BO0FBQUEsTUFNV0MsTUFOWCxjQU1XQSxNQU5YLEVBVXBDOzs7QUFDQSxNQUFNQyxtQkFBbUIsR0FBRyxTQUF0QkEsbUJBQXNCLENBQUNDLEdBQUQsRUFBY0MsT0FBZCxFQUFrQztBQUM1RGIsY0FBVSxDQUFDLEtBQUQsQ0FBVjs7QUFENEQsZ0NBRzFCYyxvRkFBMEIsQ0FDMURGLEdBQUcsQ0FBQ0csUUFBSixFQUQwRCxDQUhBO0FBQUEsUUFHcERDLFNBSG9ELHlCQUdwREEsU0FIb0Q7QUFBQSxRQUd6Q0MsVUFIeUMseUJBR3pDQSxVQUh5Qzs7QUFPNURDLCtEQUFZLENBQ1ZDLG9FQUFxQixDQUNuQjtBQUNFQyxVQUFJLEVBQUVILFVBRFI7QUFFRUksZ0JBQVUsRUFBRUwsU0FGZDtBQUdFTSxrQkFBWSxFQUFFVCxPQUhoQjtBQUlFVSxnQkFBVSxFQUFFcEIsU0FKZDtBQUtFcUIsaUJBQVcsRUFBRTtBQUxmLEtBRG1CLEVBUW5CMUIsS0FSbUIsQ0FEWCxDQUFaO0FBWUQsR0FuQkQ7O0FBcUJBLE1BQUlBLEtBQUssQ0FBQ3dCLFlBQU4sSUFBc0J4QixLQUFLLENBQUN1QixVQUFoQyxFQUE0QztBQUMxQyxXQUNFLE1BQUMsa0VBQUQ7QUFDRSxnQkFBVSxFQUFFdkIsS0FBSyxDQUFDdUIsVUFBTixJQUFvQixFQURsQztBQUVFLGtCQUFZLEVBQUV2QixLQUFLLENBQUN3QixZQUFOLElBQXNCLEVBRnRDO0FBR0UsaUJBQVcsRUFBRXhCLEtBQUssQ0FBQzJCLFdBQU4sSUFBcUIsRUFIcEM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQURGO0FBT0QsR0FSRCxNQVFPLElBQUlqQixTQUFKLEVBQWU7QUFDcEIsV0FDRSxNQUFDLDZEQUFEO0FBQ0UsZUFBUyxFQUFFQyxTQURiO0FBRUUscUJBQWUsRUFBRUYsZUFGbkI7QUFHRSxhQUFPLEVBQUVJLG1CQUhYO0FBSUUsWUFBTSxFQUFFRCxNQUpWO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFERjtBQVFELEdBVE0sQ0FVUDtBQUNBO0FBQ0E7QUFaTyxPQWFGLElBQ0hnQiwrREFBZ0IsQ0FBQ0MsWUFBakIsQ0FBOEJDLFdBQTlCLElBQ0EsQ0FBQzlCLEtBQUssQ0FBQ3dCLFlBRFAsSUFFQSxDQUFDeEIsS0FBSyxDQUFDdUIsVUFISixFQUlIO0FBQ0EsYUFBTyxNQUFDLDZFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsUUFBUDtBQUNEOztBQUNELFNBQ0UsTUFBQywyRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxvRUFBRDtBQUFhLFlBQVEsTUFBckI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsa0VBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURGLHVCQURGLENBREY7QUFRRCxDQXBFTTs7R0FBTTFCLGdCO1VBQ0lFLHFELEVBRVFFLDZFLEVBR21DSywwRDs7O0tBTi9DVCxnQiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC42OTJmMzhmOGY2YjZjMzdkMTk3MC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xuXG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vaW50ZXJmYWNlcyc7XG5pbXBvcnQgRm9sZGVyc0FuZFBsb3RzIGZyb20gJy4vZm9sZGVyc19hbmRfcGxvdHNfY29udGVudCc7XG5pbXBvcnQgeyB1c2VTZWFyY2ggfSBmcm9tICcuLi8uLi8uLi9ob29rcy91c2VTZWFyY2gnO1xuaW1wb3J0IFNlYXJjaFJlc3VsdHMgZnJvbSAnLi4vLi4vc2VhcmNoL1NlYXJjaFJlc3VsdHMnO1xuaW1wb3J0IHtcbiAgTm90Rm91bmREaXZXcmFwcGVyLFxuICBDaGFydEljb24sXG4gIE5vdEZvdW5kRGl2LFxufSBmcm9tICcuLi8uLi9zZWFyY2gvc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyBzZXBlcmF0ZVJ1bkFuZEx1bWlJblNlYXJjaCB9IGZyb20gJy4uLy4uLy4uL2NvbXBvbmVudHMvdXRpbHMnO1xuaW1wb3J0IHsgY2hhbmdlUm91dGVyLCBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMgfSBmcm9tICcuLi91dGlscyc7XG5pbXBvcnQgeyB3b3Jrc3BhY2VzIH0gZnJvbSAnLi4vLi4vLi4vd29ya3NwYWNlcy9vZmZsaW5lJztcbmltcG9ydCB7IGZ1bmN0aW9uc19jb25maWcgfSBmcm9tICcuLi8uLi8uLi9jb25maWcvY29uZmlnJztcbmltcG9ydCB7IExhdGVzdFJ1bnMgfSBmcm9tICcuLi8uLi8uLi9jb21wb25lbnRzL2luaXRpYWxQYWdlL2xhdGVzdFJ1bnMnO1xuaW1wb3J0IHsgdXNlVXBkYXRlTGl2ZU1vZGUgfSBmcm9tICcuLi8uLi8uLi9ob29rcy91c2VVcGRhdGVJbkxpdmVNb2RlJztcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vLi4vLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0JztcblxuZXhwb3J0IGNvbnN0IENvbnRlbnRTd2l0Y2hpbmcgPSAoKSA9PiB7XG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcbiAgY29uc3QgeyBzZXRfdXBkYXRlIH0gPSB1c2VVcGRhdGVMaXZlTW9kZSgpO1xuICBjb25zdCB7IHdva3JzcGFjZSB9ID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSlcblxuICBjb25zdCB7IHJlc3VsdHNfZ3JvdXBlZCwgc2VhcmNoaW5nLCBpc0xvYWRpbmcsIGVycm9ycyB9ID0gdXNlU2VhcmNoKFxuICAgIHF1ZXJ5LnNlYXJjaF9ydW5fbnVtYmVyLFxuICAgIHF1ZXJ5LnNlYXJjaF9kYXRhc2V0X25hbWUsXG4gICk7XG4gIC8vc2VyY2hSZXN1bHRzSGFuZGxlciB3aGVuIHlvdSBzZWxlY3RpbmcgcnVuLCBkYXRhc2V0IGZyb20gc2VhcmNoIHJlc3VsdHNcbiAgY29uc3Qgc2VyY2hSZXN1bHRzSGFuZGxlciA9IChydW46IHN0cmluZywgZGF0YXNldDogc3RyaW5nKSA9PiB7XG4gICAgc2V0X3VwZGF0ZShmYWxzZSk7XG5cbiAgICBjb25zdCB7IHBhcnNlZFJ1biwgcGFyc2VkTHVtaSB9ID0gc2VwZXJhdGVSdW5BbmRMdW1pSW5TZWFyY2goXG4gICAgICBydW4udG9TdHJpbmcoKVxuICAgICk7XG5cbiAgICBjaGFuZ2VSb3V0ZXIoXG4gICAgICBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMoXG4gICAgICAgIHtcbiAgICAgICAgICBsdW1pOiBwYXJzZWRMdW1pLFxuICAgICAgICAgIHJ1bl9udW1iZXI6IHBhcnNlZFJ1bixcbiAgICAgICAgICBkYXRhc2V0X25hbWU6IGRhdGFzZXQsXG4gICAgICAgICAgd29ya3NwYWNlczogd29rcnNwYWNlLFxuICAgICAgICAgIHBsb3Rfc2VhcmNoOiAnJyxcbiAgICAgICAgfSxcbiAgICAgICAgcXVlcnlcbiAgICAgIClcbiAgICApO1xuICB9O1xuXG4gIGlmIChxdWVyeS5kYXRhc2V0X25hbWUgJiYgcXVlcnkucnVuX251bWJlcikge1xuICAgIHJldHVybiAoXG4gICAgICA8Rm9sZGVyc0FuZFBsb3RzXG4gICAgICAgIHJ1bl9udW1iZXI9e3F1ZXJ5LnJ1bl9udW1iZXIgfHwgJyd9XG4gICAgICAgIGRhdGFzZXRfbmFtZT17cXVlcnkuZGF0YXNldF9uYW1lIHx8ICcnfVxuICAgICAgICBmb2xkZXJfcGF0aD17cXVlcnkuZm9sZGVyX3BhdGggfHwgJyd9XG4gICAgICAvPlxuICAgICk7XG4gIH0gZWxzZSBpZiAoc2VhcmNoaW5nKSB7XG4gICAgcmV0dXJuIChcbiAgICAgIDxTZWFyY2hSZXN1bHRzXG4gICAgICAgIGlzTG9hZGluZz17aXNMb2FkaW5nfVxuICAgICAgICByZXN1bHRzX2dyb3VwZWQ9e3Jlc3VsdHNfZ3JvdXBlZH1cbiAgICAgICAgaGFuZGxlcj17c2VyY2hSZXN1bHRzSGFuZGxlcn1cbiAgICAgICAgZXJyb3JzPXtlcnJvcnN9XG4gICAgICAvPlxuICAgICk7XG4gIH1cbiAgLy8gIXF1ZXJ5LmRhdGFzZXRfbmFtZSAmJiAhcXVlcnkucnVuX251bWJlciBiZWNhdXNlIEkgZG9uJ3Qgd2FudFxuICAvLyB0byBzZWUgbGF0ZXN0IHJ1bnMgbGlzdCwgd2hlbiBJJ20gbG9hZGluZyBmb2xkZXJzIG9yIHBsb3RzXG4gIC8vICBmb2xkZXJzIGFuZCAgcGxvdHMgYXJlIHZpc2libGUsIHdoZW4gZGF0YXNldF9uYW1lIGFuZCBydW5fbnVtYmVyIGlzIHNldFxuICBlbHNlIGlmIChcbiAgICBmdW5jdGlvbnNfY29uZmlnLm5ld19iYWNrX2VuZC5sYXRlc3RfcnVucyAmJlxuICAgICFxdWVyeS5kYXRhc2V0X25hbWUgJiZcbiAgICAhcXVlcnkucnVuX251bWJlclxuICApIHtcbiAgICByZXR1cm4gPExhdGVzdFJ1bnMgLz47XG4gIH1cbiAgcmV0dXJuIChcbiAgICA8Tm90Rm91bmREaXZXcmFwcGVyPlxuICAgICAgPE5vdEZvdW5kRGl2IG5vQm9yZGVyPlxuICAgICAgICA8Q2hhcnRJY29uIC8+XG4gICAgICAgIFdlbGNvbWUgdG8gRFFNIEdVSVxuICAgICAgPC9Ob3RGb3VuZERpdj5cbiAgICA8L05vdEZvdW5kRGl2V3JhcHBlcj5cbiAgKTtcbn07XG4iXSwic291cmNlUm9vdCI6IiJ9