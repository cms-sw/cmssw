webpackHotUpdate_N_E("pages/index",{

/***/ "./containers/display/utils.ts":
/*!*************************************!*\
  !*** ./containers/display/utils.ts ***!
  \*************************************/
/*! exports provided: getFolderPath, isPlotSelected, getSelectedPlotsNames, getSelectedPlots, getFolderPathToQuery, getContents, getDirectories, getFormatedPlotsObject, getFilteredDirectories, getChangedQueryParams, changeRouter, getNameAndDirectoriesFromDir, is_run_selected_already, choose_api, choose_api_for_run_search */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getFolderPath", function() { return getFolderPath; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "isPlotSelected", function() { return isPlotSelected; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getSelectedPlotsNames", function() { return getSelectedPlotsNames; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getSelectedPlots", function() { return getSelectedPlots; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getFolderPathToQuery", function() { return getFolderPathToQuery; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getContents", function() { return getContents; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getDirectories", function() { return getDirectories; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getFormatedPlotsObject", function() { return getFormatedPlotsObject; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getFilteredDirectories", function() { return getFilteredDirectories; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getChangedQueryParams", function() { return getChangedQueryParams; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "changeRouter", function() { return changeRouter; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getNameAndDirectoriesFromDir", function() { return getNameAndDirectoriesFromDir; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "is_run_selected_already", function() { return is_run_selected_already; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "choose_api", function() { return choose_api; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "choose_api_for_run_search", function() { return choose_api_for_run_search; });
/* harmony import */ var clean_deep__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! clean-deep */ "./node_modules/clean-deep/src/index.js");
/* harmony import */ var clean_deep__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(clean_deep__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var lodash__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! lodash */ "./node_modules/lodash/lodash.js");
/* harmony import */ var lodash__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(lodash__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var qs__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! qs */ "./node_modules/qs/lib/index.js");
/* harmony import */ var qs__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(qs__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var _components_workspaces_utils__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../components/workspaces/utils */ "./components/workspaces/utils.ts");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");






var getFolderPath = function getFolderPath(folders, clickedFolder) {
  var folderIndex = folders.indexOf(clickedFolder);
  var restFolders = folders.slice(0, folderIndex + 1);
  console.log(restFolders);
  var foldersString = restFolders.join('/');
  return foldersString;
};
var isPlotSelected = function isPlotSelected(selected_plots, plot_name) {
  return selected_plots.some(function (selected_plot) {
    return selected_plot.name === plot_name;
  });
};
var getSelectedPlotsNames = function getSelectedPlotsNames(plotsNames) {
  var plots = plotsNames ? plotsNames.split('/') : [];
  return plots;
};
var getSelectedPlots = function getSelectedPlots(plotsQuery, plots) {
  var plotsWithDirs = plotsQuery ? plotsQuery.split('&') : [];
  return plotsWithDirs.map(function (plotWithDir) {
    var plotAndDir = plotWithDir.split('/');
    var name = plotAndDir.pop();
    var directories = plotAndDir.join('/');
    var plot = plots.filter(function (plot) {
      return plot.name === name && plot.path === directories;
    });
    var displayedName = plot.length > 0 && plot[0].displayedName ? plot[0].displayedName : '';
    var qresults = plot[0] && plot[0].qresults;
    var plotObject = {
      name: name ? name : '',
      path: directories,
      displayedName: displayedName,
      qresults: qresults
    };
    return plotObject;
  });
};
var getFolderPathToQuery = function getFolderPathToQuery(previuosFolderPath, currentSelected) {
  return previuosFolderPath ? "".concat(previuosFolderPath, "/").concat(currentSelected) : "/".concat(currentSelected);
}; // what is streamerinfo? (coming from api, we don't know what it is, so we filtered it out)
// getContent also sorting data that directories should be displayed firstly, just after them- plots images.

var getContents = function getContents(data) {
  if (_config_config__WEBPACK_IMPORTED_MODULE_5__["functions_config"].new_back_end.new_back_end) {
    return data ? lodash__WEBPACK_IMPORTED_MODULE_1___default.a.sortBy(data.data ? data.data : [], ['subdir']) : [];
  }

  return data ? lodash__WEBPACK_IMPORTED_MODULE_1___default.a.sortBy(data.contents ? data.contents : [].filter(function (one_item) {
    return !one_item.hasOwnProperty('streamerinfo');
  }), ['subdir']) : [];
};
var getDirectories = function getDirectories(contents) {
  return clean_deep__WEBPACK_IMPORTED_MODULE_0___default()(contents.map(function (content) {
    if (_config_config__WEBPACK_IMPORTED_MODULE_5__["functions_config"].new_back_end.new_back_end) {
      return {
        subdir: content.subdir,
        me_count: content.me_count
      };
    }

    return {
      subdir: content.subdir
    };
  }));
};
var getFormatedPlotsObject = function getFormatedPlotsObject(contents) {
  return clean_deep__WEBPACK_IMPORTED_MODULE_0___default()(contents.map(function (content) {
    return {
      displayedName: content.obj,
      path: content.path && '/' + content.path,
      properties: content.properties
    };
  })).sort();
};
var getFilteredDirectories = function getFilteredDirectories(plot_search_folders, workspace_folders) {
  //if workspaceFolders array from context is not empty we taking intersection between all directories and workspaceFolders
  // workspace folders are fileterd folders array by selected workspace
  if (workspace_folders.length > 0) {
    var names_of_folders = plot_search_folders.map(function (folder) {
      return folder.subdir;
    }); //@ts-ignore

    var filteredDirectories = workspace_folders.filter(function (directory) {
      return directory && names_of_folders.includes(directory.subdir);
    });
    return filteredDirectories;
  } // if folder_path and workspaceFolders are empty, we return all direstories
  else if (workspace_folders.length === 0) {
      return plot_search_folders;
    }
};
var getChangedQueryParams = function getChangedQueryParams(params, query) {
  params.dataset_name = params.dataset_name ? params.dataset_name : decodeURIComponent(query.dataset_name);
  params.run_number = params.run_number ? params.run_number : query.run_number;
  params.folder_path = params.folder_path ? Object(_components_workspaces_utils__WEBPACK_IMPORTED_MODULE_4__["removeFirstSlash"])(params.folder_path) : query.folder_path;
  params.workspaces = params.workspaces ? params.workspaces : query.workspaces;
  params.overlay = params.overlay ? params.overlay : query.overlay;
  params.overlay_data = params.overlay_data === '' || params.overlay_data ? params.overlay_data : query.overlay_data;
  params.selected_plots = params.selected_plots === '' || params.selected_plots ? params.selected_plots : query.selected_plots; // if value of search field is empty string, should be retuned all folders.
  // if params.plot_search == '' when request is done, params.plot_search is changed to .*

  params.plot_search = params.plot_search === '' || params.plot_search ? params.plot_search : query.plot_search;
  params.overlay = params.overlay ? params.overlay : query.overlay;
  params.normalize = params.normalize ? params.normalize : query.normalize;
  params.lumi = params.lumi || params.lumi === 0 ? params.lumi : query.lumi; //cleaning url: if workspace is not set (it means it's empty string), it shouldn't be visible in url

  var cleaned_parameters = clean_deep__WEBPACK_IMPORTED_MODULE_0___default()(params);
  return cleaned_parameters;
};
var changeRouter = function changeRouter(parameters) {
  var queryString = qs__WEBPACK_IMPORTED_MODULE_2___default.a.stringify(parameters, {});
  next_router__WEBPACK_IMPORTED_MODULE_3___default.a.push({
    pathname: '/',
    query: parameters,
    path: decodeURIComponent(queryString)
  });
};
var getNameAndDirectoriesFromDir = function getNameAndDirectoriesFromDir(content) {
  var dir = content.path;
  var partsOfDir = dir.split('/');
  var name = partsOfDir.pop();
  var directories = partsOfDir.join('/');
  return {
    name: name,
    directories: directories
  };
};
var is_run_selected_already = function is_run_selected_already(run, query) {
  return run.run_number === query.run_number && run.dataset_name === query.dataset_name;
};
var choose_api = function choose_api(params) {
  var current_api = !_config_config__WEBPACK_IMPORTED_MODULE_5__["functions_config"].new_back_end.new_back_end ? Object(_config_config__WEBPACK_IMPORTED_MODULE_5__["get_folders_and_plots_old_api"])(params) : _config_config__WEBPACK_IMPORTED_MODULE_5__["functions_config"].mode === 'ONLINE' ? Object(_config_config__WEBPACK_IMPORTED_MODULE_5__["get_folders_and_plots_new_api_with_live_mode"])(params) : Object(_config_config__WEBPACK_IMPORTED_MODULE_5__["get_folders_and_plots_new_api"])(params);
  return current_api;
};
var choose_api_for_run_search = function choose_api_for_run_search(params) {
  var current_api = !_config_config__WEBPACK_IMPORTED_MODULE_5__["functions_config"].new_back_end.new_back_end ? Object(_config_config__WEBPACK_IMPORTED_MODULE_5__["get_run_list_by_search_old_api"])(params) : _config_config__WEBPACK_IMPORTED_MODULE_5__["functions_config"].mode === 'ONLINE' ? Object(_config_config__WEBPACK_IMPORTED_MODULE_5__["get_run_list_by_search_new_api_with_no_older_than"])(params) : Object(_config_config__WEBPACK_IMPORTED_MODULE_5__["get_run_list_by_search_new_api"])(params);
  return current_api;
};

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGFpbmVycy9kaXNwbGF5L3V0aWxzLnRzIl0sIm5hbWVzIjpbImdldEZvbGRlclBhdGgiLCJmb2xkZXJzIiwiY2xpY2tlZEZvbGRlciIsImZvbGRlckluZGV4IiwiaW5kZXhPZiIsInJlc3RGb2xkZXJzIiwic2xpY2UiLCJjb25zb2xlIiwibG9nIiwiZm9sZGVyc1N0cmluZyIsImpvaW4iLCJpc1Bsb3RTZWxlY3RlZCIsInNlbGVjdGVkX3Bsb3RzIiwicGxvdF9uYW1lIiwic29tZSIsInNlbGVjdGVkX3Bsb3QiLCJuYW1lIiwiZ2V0U2VsZWN0ZWRQbG90c05hbWVzIiwicGxvdHNOYW1lcyIsInBsb3RzIiwic3BsaXQiLCJnZXRTZWxlY3RlZFBsb3RzIiwicGxvdHNRdWVyeSIsInBsb3RzV2l0aERpcnMiLCJtYXAiLCJwbG90V2l0aERpciIsInBsb3RBbmREaXIiLCJwb3AiLCJkaXJlY3RvcmllcyIsInBsb3QiLCJmaWx0ZXIiLCJwYXRoIiwiZGlzcGxheWVkTmFtZSIsImxlbmd0aCIsInFyZXN1bHRzIiwicGxvdE9iamVjdCIsImdldEZvbGRlclBhdGhUb1F1ZXJ5IiwicHJldml1b3NGb2xkZXJQYXRoIiwiY3VycmVudFNlbGVjdGVkIiwiZ2V0Q29udGVudHMiLCJkYXRhIiwiZnVuY3Rpb25zX2NvbmZpZyIsIm5ld19iYWNrX2VuZCIsIl8iLCJzb3J0QnkiLCJjb250ZW50cyIsIm9uZV9pdGVtIiwiaGFzT3duUHJvcGVydHkiLCJnZXREaXJlY3RvcmllcyIsImNsZWFuRGVlcCIsImNvbnRlbnQiLCJzdWJkaXIiLCJtZV9jb3VudCIsImdldEZvcm1hdGVkUGxvdHNPYmplY3QiLCJvYmoiLCJwcm9wZXJ0aWVzIiwic29ydCIsImdldEZpbHRlcmVkRGlyZWN0b3JpZXMiLCJwbG90X3NlYXJjaF9mb2xkZXJzIiwid29ya3NwYWNlX2ZvbGRlcnMiLCJuYW1lc19vZl9mb2xkZXJzIiwiZm9sZGVyIiwiZmlsdGVyZWREaXJlY3RvcmllcyIsImRpcmVjdG9yeSIsImluY2x1ZGVzIiwiZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zIiwicGFyYW1zIiwicXVlcnkiLCJkYXRhc2V0X25hbWUiLCJkZWNvZGVVUklDb21wb25lbnQiLCJydW5fbnVtYmVyIiwiZm9sZGVyX3BhdGgiLCJyZW1vdmVGaXJzdFNsYXNoIiwid29ya3NwYWNlcyIsIm92ZXJsYXkiLCJvdmVybGF5X2RhdGEiLCJwbG90X3NlYXJjaCIsIm5vcm1hbGl6ZSIsImx1bWkiLCJjbGVhbmVkX3BhcmFtZXRlcnMiLCJjaGFuZ2VSb3V0ZXIiLCJwYXJhbWV0ZXJzIiwicXVlcnlTdHJpbmciLCJxcyIsInN0cmluZ2lmeSIsIlJvdXRlciIsInB1c2giLCJwYXRobmFtZSIsImdldE5hbWVBbmREaXJlY3Rvcmllc0Zyb21EaXIiLCJkaXIiLCJwYXJ0c09mRGlyIiwiaXNfcnVuX3NlbGVjdGVkX2FscmVhZHkiLCJydW4iLCJjaG9vc2VfYXBpIiwiY3VycmVudF9hcGkiLCJnZXRfZm9sZGVyc19hbmRfcGxvdHNfb2xkX2FwaSIsIm1vZGUiLCJnZXRfZm9sZGVyc19hbmRfcGxvdHNfbmV3X2FwaV93aXRoX2xpdmVfbW9kZSIsImdldF9mb2xkZXJzX2FuZF9wbG90c19uZXdfYXBpIiwiY2hvb3NlX2FwaV9mb3JfcnVuX3NlYXJjaCIsImdldF9ydW5fbGlzdF9ieV9zZWFyY2hfb2xkX2FwaSIsImdldF9ydW5fbGlzdF9ieV9zZWFyY2hfbmV3X2FwaV93aXRoX25vX29sZGVyX3RoYW4iLCJnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX25ld19hcGkiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBO0FBU0E7QUFFQTtBQUNBO0FBVU8sSUFBTUEsYUFB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QXpCLElBQ0FzQixHQUFHLENBQUN4QixZQUFKLEtBQXFCRCxLQUFLLENBQUNDLFlBRjdCO0FBSUQsQ0FSTTtBQVVBLElBQU15QixVQUFVLEdBQUcsU0FBYkEsVUFBYSxDQUFDM0IsTUFBRCxFQUErQjtBQUN2RCxNQUFNNEIsV0FBVyxHQUFHLENBQUNyRCwrREFBZ0IsQ0FBQ0MsWUFBakIsQ0FBOEJBLFlBQS9CLEdBQ2hCcUQsb0ZBQTZCLENBQUM3QixNQUFELENBRGIsR0FFaEJ6QiwrREFBZ0IsQ0FBQ3VELElBQWpCLEtBQTBCLFFBQTFCLEdBQ0FDLG1HQUE0QyxDQUFDL0IsTUFBRCxDQUQ1QyxHQUVBZ0Msb0ZBQTZCLENBQUNoQyxNQUFELENBSmpDO0FBS0EsU0FBTzRCLFdBQVA7QUFDRCxDQVBNO0FBU0EsSUFBTUsseUJBQXlCLEdBQUcsU0FBNUJBLHlCQUE0QixDQUFDakMsTUFBRCxFQUErQjtBQUN0RSxNQUFNNEIsV0FBVyxHQUFHLENBQUNyRCwrREFBZ0IsQ0FBQ0MsWUFBakIsQ0FBOEJBLFlBQS9CLEdBQ2hCMEQscUZBQThCLENBQUNsQyxNQUFELENBRGQsR0FFaEJ6QiwrREFBZ0IsQ0FBQ3VELElBQWpCLEtBQTBCLFFBQTFCLEdBQ0FLLHdHQUFpRCxDQUFDbkMsTUFBRCxDQURqRCxHQUVBb0MscUZBQThCLENBQUNwQyxNQUFELENBSmxDO0FBTUEsU0FBTzRCLFdBQVA7QUFDRCxDQVJNIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmY4MzI2ZDc1YmExNDIwMzI2ZWE5LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgY2xlYW5EZWVwIGZyb20gJ2NsZWFuLWRlZXAnO1xuaW1wb3J0IF8gZnJvbSAnbG9kYXNoJztcbmltcG9ydCBxcyBmcm9tICdxcyc7XG5cbmltcG9ydCB7XG4gIFBsb3REYXRhUHJvcHMsXG4gIFBsb3RJbnRlcmZhY2UsXG4gIERpcmVjdG9yeUludGVyZmFjZSxcbiAgUXVlcnlQcm9wcyxcbiAgUGFyYW1zRm9yQXBpUHJvcHMsXG59IGZyb20gJy4vaW50ZXJmYWNlcyc7XG5pbXBvcnQgUm91dGVyIGZyb20gJ25leHQvcm91dGVyJztcbmltcG9ydCB7IFBhcnNlZFVybFF1ZXJ5SW5wdXQgfSBmcm9tICdxdWVyeXN0cmluZyc7XG5pbXBvcnQgeyByZW1vdmVGaXJzdFNsYXNoIH0gZnJvbSAnLi4vLi4vY29tcG9uZW50cy93b3Jrc3BhY2VzL3V0aWxzJztcbmltcG9ydCB7XG4gIGZ1bmN0aW9uc19jb25maWcsXG4gIGdldF9mb2xkZXJzX2FuZF9wbG90c19vbGRfYXBpLFxuICBnZXRfZm9sZGVyc19hbmRfcGxvdHNfbmV3X2FwaSxcbiAgZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX25ld19hcGlfd2l0aF9saXZlX21vZGUsXG4gIGdldF9ydW5fbGlzdF9ieV9zZWFyY2hfbmV3X2FwaSxcbiAgZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9vbGRfYXBpLFxuICBnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX25ld19hcGlfd2l0aF9ub19vbGRlcl90aGFuLFxufSBmcm9tICcuLi8uLi9jb25maWcvY29uZmlnJztcblxuZXhwb3J0IGNvbnN0IGdldEZvbGRlclBhdGggPSAoZm9sZGVyczogc3RyaW5nW10sIGNsaWNrZWRGb2xkZXI6IHN0cmluZykgPT4ge1xuICBjb25zdCBmb2xkZXJJbmRleCA9IGZvbGRlcnMuaW5kZXhPZihjbGlja2VkRm9sZGVyKTtcbiAgY29uc3QgcmVzdEZvbGRlcnM6IHN0cmluZ1tdID0gZm9sZGVycy5zbGljZSgwLCBmb2xkZXJJbmRleCArIDEpO1xuICBjb25zb2xlLmxvZyhyZXN0Rm9sZGVycylcbiAgY29uc3QgZm9sZGVyc1N0cmluZyA9IHJlc3RGb2xkZXJzLmpvaW4oJy8nKTtcbiAgcmV0dXJuIGZvbGRlcnNTdHJpbmc7XG59O1xuXG5leHBvcnQgY29uc3QgaXNQbG90U2VsZWN0ZWQgPSAoXG4gIHNlbGVjdGVkX3Bsb3RzOiBQbG90RGF0YVByb3BzW10sXG4gIHBsb3RfbmFtZTogc3RyaW5nXG4pID0+XG4gIHNlbGVjdGVkX3Bsb3RzLnNvbWUoXG4gICAgKHNlbGVjdGVkX3Bsb3Q6IFBsb3REYXRhUHJvcHMpID0+IHNlbGVjdGVkX3Bsb3QubmFtZSA9PT0gcGxvdF9uYW1lXG4gICk7XG5cbmV4cG9ydCBjb25zdCBnZXRTZWxlY3RlZFBsb3RzTmFtZXMgPSAocGxvdHNOYW1lczogc3RyaW5nIHwgdW5kZWZpbmVkKSA9PiB7XG4gIGNvbnN0IHBsb3RzID0gcGxvdHNOYW1lcyA/IHBsb3RzTmFtZXMuc3BsaXQoJy8nKSA6IFtdO1xuXG4gIHJldHVybiBwbG90cztcbn07XG5cbmV4cG9ydCBjb25zdCBnZXRTZWxlY3RlZFBsb3RzID0gKFxuICBwbG90c1F1ZXJ5OiBzdHJpbmcgfCB1bmRlZmluZWQsXG4gIHBsb3RzOiBQbG90RGF0YVByb3BzW11cbikgPT4ge1xuICBjb25zdCBwbG90c1dpdGhEaXJzID0gcGxvdHNRdWVyeSA/IHBsb3RzUXVlcnkuc3BsaXQoJyYnKSA6IFtdO1xuICByZXR1cm4gcGxvdHNXaXRoRGlycy5tYXAoKHBsb3RXaXRoRGlyOiBzdHJpbmcpID0+IHtcbiAgICBjb25zdCBwbG90QW5kRGlyID0gcGxvdFdpdGhEaXIuc3BsaXQoJy8nKTtcbiAgICBjb25zdCBuYW1lID0gcGxvdEFuZERpci5wb3AoKTtcbiAgICBjb25zdCBkaXJlY3RvcmllcyA9IHBsb3RBbmREaXIuam9pbignLycpO1xuICAgIGNvbnN0IHBsb3QgPSBwbG90cy5maWx0ZXIoXG4gICAgICAocGxvdCkgPT4gcGxvdC5uYW1lID09PSBuYW1lICYmIHBsb3QucGF0aCA9PT0gZGlyZWN0b3JpZXNcbiAgICApO1xuICAgIGNvbnN0IGRpc3BsYXllZE5hbWUgPVxuICAgICAgcGxvdC5sZW5ndGggPiAwICYmIHBsb3RbMF0uZGlzcGxheWVkTmFtZSA/IHBsb3RbMF0uZGlzcGxheWVkTmFtZSA6ICcnO1xuXG4gICAgY29uc3QgcXJlc3VsdHMgPSBwbG90WzBdICYmIHBsb3RbMF0ucXJlc3VsdHM7XG5cbiAgICBjb25zdCBwbG90T2JqZWN0OiBQbG90RGF0YVByb3BzID0ge1xuICAgICAgbmFtZTogbmFtZSA/IG5hbWUgOiAnJyxcbiAgICAgIHBhdGg6IGRpcmVjdG9yaWVzLFxuICAgICAgZGlzcGxheWVkTmFtZTogZGlzcGxheWVkTmFtZSxcbiAgICAgIHFyZXN1bHRzOiBxcmVzdWx0cyxcbiAgICB9O1xuICAgIHJldHVybiBwbG90T2JqZWN0O1xuICB9KTtcbn07XG5cbmV4cG9ydCBjb25zdCBnZXRGb2xkZXJQYXRoVG9RdWVyeSA9IChcbiAgcHJldml1b3NGb2xkZXJQYXRoOiBzdHJpbmcgfCB1bmRlZmluZWQsXG4gIGN1cnJlbnRTZWxlY3RlZDogc3RyaW5nXG4pID0+IHtcbiAgcmV0dXJuIHByZXZpdW9zRm9sZGVyUGF0aFxuICAgID8gYCR7cHJldml1b3NGb2xkZXJQYXRofS8ke2N1cnJlbnRTZWxlY3RlZH1gXG4gICAgOiBgLyR7Y3VycmVudFNlbGVjdGVkfWA7XG59O1xuXG4vLyB3aGF0IGlzIHN0cmVhbWVyaW5mbz8gKGNvbWluZyBmcm9tIGFwaSwgd2UgZG9uJ3Qga25vdyB3aGF0IGl0IGlzLCBzbyB3ZSBmaWx0ZXJlZCBpdCBvdXQpXG4vLyBnZXRDb250ZW50IGFsc28gc29ydGluZyBkYXRhIHRoYXQgZGlyZWN0b3JpZXMgc2hvdWxkIGJlIGRpc3BsYXllZCBmaXJzdGx5LCBqdXN0IGFmdGVyIHRoZW0tIHBsb3RzIGltYWdlcy5cbmV4cG9ydCBjb25zdCBnZXRDb250ZW50cyA9IChkYXRhOiBhbnkpID0+IHtcbiAgaWYgKGZ1bmN0aW9uc19jb25maWcubmV3X2JhY2tfZW5kLm5ld19iYWNrX2VuZCkge1xuICAgIHJldHVybiBkYXRhID8gXy5zb3J0QnkoZGF0YS5kYXRhID8gZGF0YS5kYXRhIDogW10sIFsnc3ViZGlyJ10pIDogW107XG4gIH1cbiAgcmV0dXJuIGRhdGFcbiAgICA/IF8uc29ydEJ5KFxuICAgICAgICBkYXRhLmNvbnRlbnRzXG4gICAgICAgICAgPyBkYXRhLmNvbnRlbnRzXG4gICAgICAgICAgOiBbXS5maWx0ZXIoXG4gICAgICAgICAgICAgIChvbmVfaXRlbTogUGxvdEludGVyZmFjZSB8IERpcmVjdG9yeUludGVyZmFjZSkgPT5cbiAgICAgICAgICAgICAgICAhb25lX2l0ZW0uaGFzT3duUHJvcGVydHkoJ3N0cmVhbWVyaW5mbycpXG4gICAgICAgICAgICApLFxuICAgICAgICBbJ3N1YmRpciddXG4gICAgICApXG4gICAgOiBbXTtcbn07XG5cbmV4cG9ydCBjb25zdCBnZXREaXJlY3RvcmllczogYW55ID0gKGNvbnRlbnRzOiBEaXJlY3RvcnlJbnRlcmZhY2VbXSkgPT4ge1xuICByZXR1cm4gY2xlYW5EZWVwKFxuICAgIGNvbnRlbnRzLm1hcCgoY29udGVudDogRGlyZWN0b3J5SW50ZXJmYWNlKSA9PiB7XG4gICAgICBpZiAoZnVuY3Rpb25zX2NvbmZpZy5uZXdfYmFja19lbmQubmV3X2JhY2tfZW5kKSB7XG4gICAgICAgIHJldHVybiB7IHN1YmRpcjogY29udGVudC5zdWJkaXIsIG1lX2NvdW50OiBjb250ZW50Lm1lX2NvdW50IH07XG4gICAgICB9XG4gICAgICByZXR1cm4geyBzdWJkaXI6IGNvbnRlbnQuc3ViZGlyIH07XG4gICAgfSlcbiAgKTtcbn07XG5cbmV4cG9ydCBjb25zdCBnZXRGb3JtYXRlZFBsb3RzT2JqZWN0ID0gKGNvbnRlbnRzOiBQbG90SW50ZXJmYWNlW10pID0+XG4gIGNsZWFuRGVlcChcbiAgICBjb250ZW50cy5tYXAoKGNvbnRlbnQ6IFBsb3RJbnRlcmZhY2UpID0+IHtcbiAgICAgIHJldHVybiB7XG4gICAgICAgIGRpc3BsYXllZE5hbWU6IGNvbnRlbnQub2JqLFxuICAgICAgICBwYXRoOiBjb250ZW50LnBhdGggJiYgJy8nICsgY29udGVudC5wYXRoLFxuICAgICAgICBwcm9wZXJ0aWVzOiBjb250ZW50LnByb3BlcnRpZXMsXG4gICAgICB9O1xuICAgIH0pXG4gICkuc29ydCgpO1xuXG5leHBvcnQgY29uc3QgZ2V0RmlsdGVyZWREaXJlY3RvcmllcyA9IChcbiAgcGxvdF9zZWFyY2hfZm9sZGVyczogRGlyZWN0b3J5SW50ZXJmYWNlW10sXG4gIHdvcmtzcGFjZV9mb2xkZXJzOiAoRGlyZWN0b3J5SW50ZXJmYWNlIHwgdW5kZWZpbmVkKVtdXG4pID0+IHtcbiAgLy9pZiB3b3Jrc3BhY2VGb2xkZXJzIGFycmF5IGZyb20gY29udGV4dCBpcyBub3QgZW1wdHkgd2UgdGFraW5nIGludGVyc2VjdGlvbiBiZXR3ZWVuIGFsbCBkaXJlY3RvcmllcyBhbmQgd29ya3NwYWNlRm9sZGVyc1xuICAvLyB3b3Jrc3BhY2UgZm9sZGVycyBhcmUgZmlsZXRlcmQgZm9sZGVycyBhcnJheSBieSBzZWxlY3RlZCB3b3Jrc3BhY2VcbiAgaWYgKHdvcmtzcGFjZV9mb2xkZXJzLmxlbmd0aCA+IDApIHtcbiAgICBjb25zdCBuYW1lc19vZl9mb2xkZXJzID0gcGxvdF9zZWFyY2hfZm9sZGVycy5tYXAoXG4gICAgICAoZm9sZGVyOiBEaXJlY3RvcnlJbnRlcmZhY2UpID0+IGZvbGRlci5zdWJkaXJcbiAgICApO1xuICAgIC8vQHRzLWlnbm9yZVxuICAgIGNvbnN0IGZpbHRlcmVkRGlyZWN0b3JpZXMgPSB3b3Jrc3BhY2VfZm9sZGVycy5maWx0ZXIoXG4gICAgICAoZGlyZWN0b3J5OiBEaXJlY3RvcnlJbnRlcmZhY2UgfCB1bmRlZmluZWQpID0+XG4gICAgICAgIGRpcmVjdG9yeSAmJiBuYW1lc19vZl9mb2xkZXJzLmluY2x1ZGVzKGRpcmVjdG9yeS5zdWJkaXIpXG4gICAgKTtcbiAgICByZXR1cm4gZmlsdGVyZWREaXJlY3RvcmllcztcbiAgfVxuICAvLyBpZiBmb2xkZXJfcGF0aCBhbmQgd29ya3NwYWNlRm9sZGVycyBhcmUgZW1wdHksIHdlIHJldHVybiBhbGwgZGlyZXN0b3JpZXNcbiAgZWxzZSBpZiAod29ya3NwYWNlX2ZvbGRlcnMubGVuZ3RoID09PSAwKSB7XG4gICAgcmV0dXJuIHBsb3Rfc2VhcmNoX2ZvbGRlcnM7XG4gIH1cbn07XG5cbmV4cG9ydCBjb25zdCBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMgPSAoXG4gIHBhcmFtczogUGFyc2VkVXJsUXVlcnlJbnB1dCxcbiAgcXVlcnk6IFF1ZXJ5UHJvcHNcbikgPT4ge1xuICBwYXJhbXMuZGF0YXNldF9uYW1lID0gcGFyYW1zLmRhdGFzZXRfbmFtZVxuICAgID8gcGFyYW1zLmRhdGFzZXRfbmFtZVxuICAgIDogZGVjb2RlVVJJQ29tcG9uZW50KHF1ZXJ5LmRhdGFzZXRfbmFtZSBhcyBzdHJpbmcpO1xuXG4gIHBhcmFtcy5ydW5fbnVtYmVyID0gcGFyYW1zLnJ1bl9udW1iZXIgPyBwYXJhbXMucnVuX251bWJlciA6IHF1ZXJ5LnJ1bl9udW1iZXI7XG5cbiAgcGFyYW1zLmZvbGRlcl9wYXRoID0gcGFyYW1zLmZvbGRlcl9wYXRoXG4gICAgPyByZW1vdmVGaXJzdFNsYXNoKHBhcmFtcy5mb2xkZXJfcGF0aCBhcyBzdHJpbmcpXG4gICAgOiBxdWVyeS5mb2xkZXJfcGF0aDtcblxuICBwYXJhbXMud29ya3NwYWNlcyA9IHBhcmFtcy53b3Jrc3BhY2VzID8gcGFyYW1zLndvcmtzcGFjZXMgOiBxdWVyeS53b3Jrc3BhY2VzO1xuXG4gIHBhcmFtcy5vdmVybGF5ID0gcGFyYW1zLm92ZXJsYXkgPyBwYXJhbXMub3ZlcmxheSA6IHF1ZXJ5Lm92ZXJsYXk7XG5cbiAgcGFyYW1zLm92ZXJsYXlfZGF0YSA9XG4gICAgcGFyYW1zLm92ZXJsYXlfZGF0YSA9PT0gJycgfHwgcGFyYW1zLm92ZXJsYXlfZGF0YVxuICAgICAgPyBwYXJhbXMub3ZlcmxheV9kYXRhXG4gICAgICA6IHF1ZXJ5Lm92ZXJsYXlfZGF0YTtcblxuICBwYXJhbXMuc2VsZWN0ZWRfcGxvdHMgPVxuICAgIHBhcmFtcy5zZWxlY3RlZF9wbG90cyA9PT0gJycgfHwgcGFyYW1zLnNlbGVjdGVkX3Bsb3RzXG4gICAgICA/IHBhcmFtcy5zZWxlY3RlZF9wbG90c1xuICAgICAgOiBxdWVyeS5zZWxlY3RlZF9wbG90cztcblxuICAvLyBpZiB2YWx1ZSBvZiBzZWFyY2ggZmllbGQgaXMgZW1wdHkgc3RyaW5nLCBzaG91bGQgYmUgcmV0dW5lZCBhbGwgZm9sZGVycy5cbiAgLy8gaWYgcGFyYW1zLnBsb3Rfc2VhcmNoID09ICcnIHdoZW4gcmVxdWVzdCBpcyBkb25lLCBwYXJhbXMucGxvdF9zZWFyY2ggaXMgY2hhbmdlZCB0byAuKlxuICBwYXJhbXMucGxvdF9zZWFyY2ggPVxuICAgIHBhcmFtcy5wbG90X3NlYXJjaCA9PT0gJycgfHwgcGFyYW1zLnBsb3Rfc2VhcmNoXG4gICAgICA/IHBhcmFtcy5wbG90X3NlYXJjaFxuICAgICAgOiBxdWVyeS5wbG90X3NlYXJjaDtcblxuICBwYXJhbXMub3ZlcmxheSA9IHBhcmFtcy5vdmVybGF5ID8gcGFyYW1zLm92ZXJsYXkgOiBxdWVyeS5vdmVybGF5O1xuXG4gIHBhcmFtcy5ub3JtYWxpemUgPSBwYXJhbXMubm9ybWFsaXplID8gcGFyYW1zLm5vcm1hbGl6ZSA6IHF1ZXJ5Lm5vcm1hbGl6ZTtcblxuICBwYXJhbXMubHVtaSA9IHBhcmFtcy5sdW1pIHx8IHBhcmFtcy5sdW1pID09PSAwID8gcGFyYW1zLmx1bWkgOiBxdWVyeS5sdW1pO1xuXG4gIC8vY2xlYW5pbmcgdXJsOiBpZiB3b3Jrc3BhY2UgaXMgbm90IHNldCAoaXQgbWVhbnMgaXQncyBlbXB0eSBzdHJpbmcpLCBpdCBzaG91bGRuJ3QgYmUgdmlzaWJsZSBpbiB1cmxcbiAgY29uc3QgY2xlYW5lZF9wYXJhbWV0ZXJzID0gY2xlYW5EZWVwKHBhcmFtcyk7XG5cbiAgcmV0dXJuIGNsZWFuZWRfcGFyYW1ldGVycztcbn07XG5cbmV4cG9ydCBjb25zdCBjaGFuZ2VSb3V0ZXIgPSAocGFyYW1ldGVyczogUGFyc2VkVXJsUXVlcnlJbnB1dCkgPT4ge1xuICBjb25zdCBxdWVyeVN0cmluZyA9IHFzLnN0cmluZ2lmeShwYXJhbWV0ZXJzLCB7fSk7XG4gIFJvdXRlci5wdXNoKHtcbiAgICBwYXRobmFtZTogJy8nLFxuICAgIHF1ZXJ5OiBwYXJhbWV0ZXJzLFxuICAgIHBhdGg6IGRlY29kZVVSSUNvbXBvbmVudChxdWVyeVN0cmluZyksXG4gIH0pO1xufTtcblxuZXhwb3J0IGNvbnN0IGdldE5hbWVBbmREaXJlY3Rvcmllc0Zyb21EaXIgPSAoY29udGVudDogUGxvdEludGVyZmFjZSkgPT4ge1xuICBjb25zdCBkaXIgPSBjb250ZW50LnBhdGg7XG4gIGNvbnN0IHBhcnRzT2ZEaXIgPSBkaXIuc3BsaXQoJy8nKTtcbiAgY29uc3QgbmFtZSA9IHBhcnRzT2ZEaXIucG9wKCk7XG4gIGNvbnN0IGRpcmVjdG9yaWVzID0gcGFydHNPZkRpci5qb2luKCcvJyk7XG5cbiAgcmV0dXJuIHsgbmFtZSwgZGlyZWN0b3JpZXMgfTtcbn07XG5cbmV4cG9ydCBjb25zdCBpc19ydW5fc2VsZWN0ZWRfYWxyZWFkeSA9IChcbiAgcnVuOiB7IHJ1bl9udW1iZXI6IHN0cmluZzsgZGF0YXNldF9uYW1lOiBzdHJpbmcgfSxcbiAgcXVlcnk6IFF1ZXJ5UHJvcHNcbikgPT4ge1xuICByZXR1cm4gKFxuICAgIHJ1bi5ydW5fbnVtYmVyID09PSBxdWVyeS5ydW5fbnVtYmVyICYmXG4gICAgcnVuLmRhdGFzZXRfbmFtZSA9PT0gcXVlcnkuZGF0YXNldF9uYW1lXG4gICk7XG59O1xuXG5leHBvcnQgY29uc3QgY2hvb3NlX2FwaSA9IChwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzKSA9PiB7XG4gIGNvbnN0IGN1cnJlbnRfYXBpID0gIWZ1bmN0aW9uc19jb25maWcubmV3X2JhY2tfZW5kLm5ld19iYWNrX2VuZFxuICAgID8gZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX29sZF9hcGkocGFyYW1zKVxuICAgIDogZnVuY3Rpb25zX2NvbmZpZy5tb2RlID09PSAnT05MSU5FJ1xuICAgID8gZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX25ld19hcGlfd2l0aF9saXZlX21vZGUocGFyYW1zKVxuICAgIDogZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX25ld19hcGkocGFyYW1zKTtcbiAgcmV0dXJuIGN1cnJlbnRfYXBpO1xufTtcblxuZXhwb3J0IGNvbnN0IGNob29zZV9hcGlfZm9yX3J1bl9zZWFyY2ggPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcykgPT4ge1xuICBjb25zdCBjdXJyZW50X2FwaSA9ICFmdW5jdGlvbnNfY29uZmlnLm5ld19iYWNrX2VuZC5uZXdfYmFja19lbmRcbiAgICA/IGdldF9ydW5fbGlzdF9ieV9zZWFyY2hfb2xkX2FwaShwYXJhbXMpXG4gICAgOiBmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnXG4gICAgPyBnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX25ld19hcGlfd2l0aF9ub19vbGRlcl90aGFuKHBhcmFtcylcbiAgICA6IGdldF9ydW5fbGlzdF9ieV9zZWFyY2hfbmV3X2FwaShwYXJhbXMpO1xuXG4gIHJldHVybiBjdXJyZW50X2FwaTtcbn07XG4iXSwic291cmNlUm9vdCI6IiJ9