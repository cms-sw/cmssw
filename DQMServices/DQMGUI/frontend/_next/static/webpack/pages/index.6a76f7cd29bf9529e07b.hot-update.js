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
    pathname: "/",
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGFpbmVycy9kaXNwbGF5L3V0aWxzLnRzIl0sIm5hbWVzIjpbImdldEZvbGRlclBhdGgiLCJmb2xkZXJzIiwiY2xpY2tlZEZvbGRlciIsImZvbGRlckluZGV4IiwiaW5kZXhPZiIsInJlc3RGb2xkZXJzIiwic2xpY2UiLCJmb2xkZXJzU3RyaW5nIiwiam9pbiIsImlzUGxvdFNlbGVjdGVkIiwic2VsZWN0ZWRfcGxvdHMiLCJwbG90X25hbWUiLCJzb21lIiwic2VsZWN0ZWRfcGxvdCIsIm5hbWUiLCJnZXRTZWxlY3RlZFBsb3RzTmFtZXMiLCJwbG90c05hbWVzIiwicGxvdHMiLCJzcGxpdCIsImdldFNlbGVjdGVkUGxvdHMiLCJwbG90c1F1ZXJ5IiwicGxvdHNXaXRoRGlycyIsIm1hcCIsInBsb3RXaXRoRGlyIiwicGxvdEFuZERpciIsInBvcCIsImRpcmVjdG9yaWVzIiwicGxvdCIsImZpbHRlciIsInBhdGgiLCJkaXNwbGF5ZWROYW1lIiwibGVuZ3RoIiwicXJlc3VsdHMiLCJwbG90T2JqZWN0IiwiZ2V0Rm9sZGVyUGF0aFRvUXVlcnkiLCJwcmV2aXVvc0ZvbGRlclBhdGgiLCJjdXJyZW50U2VsZWN0ZWQiLCJnZXRDb250ZW50cyIsImRhdGEiLCJmdW5jdGlvbnNfY29uZmlnIiwibmV3X2JhY2tfZW5kIiwiXyIsInNvcnRCeSIsImNvbnRlbnRzIiwib25lX2l0ZW0iLCJoYXNPd25Qcm9wZXJ0eSIsImdldERpcmVjdG9yaWVzIiwiY2xlYW5EZWVwIiwiY29udGVudCIsInN1YmRpciIsIm1lX2NvdW50IiwiZ2V0Rm9ybWF0ZWRQbG90c09iamVjdCIsIm9iaiIsInByb3BlcnRpZXMiLCJzb3J0IiwiZ2V0RmlsdGVyZWREaXJlY3RvcmllcyIsInBsb3Rfc2VhcmNoX2ZvbGRlcnMiLCJ3b3Jrc3BhY2VfZm9sZGVycyIsIm5hbWVzX29mX2ZvbGRlcnMiLCJmb2xkZXIiLCJmaWx0ZXJlZERpcmVjdG9yaWVzIiwiZGlyZWN0b3J5IiwiaW5jbHVkZXMiLCJnZXRDaGFuZ2VkUXVlcnlQYXJhbXMiLCJwYXJhbXMiLCJxdWVyeSIsImRhdGFzZXRfbmFtZSIsImRlY29kZVVSSUNvbXBvbmVudCIsInJ1bl9udW1iZXIiLCJmb2xkZXJfcGF0aCIsInJlbW92ZUZpcnN0U2xhc2giLCJ3b3Jrc3BhY2VzIiwib3ZlcmxheSIsIm92ZXJsYXlfZGF0YSIsInBsb3Rfc2VhcmNoIiwibm9ybWFsaXplIiwibHVtaSIsImNsZWFuZWRfcGFyYW1ldGVycyIsImNoYW5nZVJvdXRlciIsInBhcmFtZXRlcnMiLCJxdWVyeVN0cmluZyIsInFzIiwic3RyaW5naWZ5IiwiUm91dGVyIiwicHVzaCIsInBhdGhuYW1lIiwiZ2V0TmFtZUFuZERpcmVjdG9yaWVzRnJvbURpciIsImRpciIsInBhcnRzT2ZEaXIiLCJpc19ydW5fc2VsZWN0ZWRfYWxyZWFkeSIsInJ1biIsImNob29zZV9hcGkiLCJjdXJyZW50X2FwaSIsImdldF9mb2xkZXJzX2FuZF9wbG90c19vbGRfYXBpIiwibW9kZSIsImdldF9mb2xkZXJzX2FuZF9wbG90c19uZXdfYXBpX3dpdGhfbGl2ZV9tb2RlIiwiZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX25ld19hcGkiLCJjaG9vc2VfYXBpX2Zvcl9ydW5fc2VhcmNoIiwiZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9vbGRfYXBpIiwiZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9uZXdfYXBpX3dpdGhfbm9fb2xkZXJfdGhhbiIsImdldF9ydW5fbGlzdF9ieV9zZWFyY2hfbmV3X2FwaSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7OztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUNBO0FBQ0E7QUFTQTtBQUVBO0FBQ0E7QUFVTyxJQUFNQSxhQUFhLEdBQUcsU0FBaEJBLGFB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SUQsQ0FSTTtBQVVBLElBQU15QixVQUFVLEdBQUcsU0FBYkEsVUFBYSxDQUFDM0IsTUFBRCxFQUErQjtBQUN2RCxNQUFNNEIsV0FBVyxHQUFHLENBQUNyRCwrREFBZ0IsQ0FBQ0MsWUFBakIsQ0FBOEJBLFlBQS9CLEdBQ2hCcUQsb0ZBQTZCLENBQUM3QixNQUFELENBRGIsR0FFaEJ6QiwrREFBZ0IsQ0FBQ3VELElBQWpCLEtBQTBCLFFBQTFCLEdBQ0FDLG1HQUE0QyxDQUFDL0IsTUFBRCxDQUQ1QyxHQUVBZ0Msb0ZBQTZCLENBQUNoQyxNQUFELENBSmpDO0FBS0EsU0FBTzRCLFdBQVA7QUFDRCxDQVBNO0FBU0EsSUFBTUsseUJBQXlCLEdBQUcsU0FBNUJBLHlCQUE0QixDQUFDakMsTUFBRCxFQUErQjtBQUN0RSxNQUFNNEIsV0FBVyxHQUFHLENBQUNyRCwrREFBZ0IsQ0FBQ0MsWUFBakIsQ0FBOEJBLFlBQS9CLEdBQ2hCMEQscUZBQThCLENBQUNsQyxNQUFELENBRGQsR0FFaEJ6QiwrREFBZ0IsQ0FBQ3VELElBQWpCLEtBQTBCLFFBQTFCLEdBQ0FLLHdHQUFpRCxDQUFDbkMsTUFBRCxDQURqRCxHQUVBb0MscUZBQThCLENBQUNwQyxNQUFELENBSmxDO0FBTUEsU0FBTzRCLFdBQVA7QUFDRCxDQVJNIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjZhNzZmN2NkMjliZjk1MjllMDdiLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgY2xlYW5EZWVwIGZyb20gJ2NsZWFuLWRlZXAnO1xuaW1wb3J0IF8gZnJvbSAnbG9kYXNoJztcbmltcG9ydCBxcyBmcm9tICdxcyc7XG5cbmltcG9ydCB7XG4gIFBsb3REYXRhUHJvcHMsXG4gIFBsb3RJbnRlcmZhY2UsXG4gIERpcmVjdG9yeUludGVyZmFjZSxcbiAgUXVlcnlQcm9wcyxcbiAgUGFyYW1zRm9yQXBpUHJvcHMsXG59IGZyb20gJy4vaW50ZXJmYWNlcyc7XG5pbXBvcnQgUm91dGVyIGZyb20gJ25leHQvcm91dGVyJztcbmltcG9ydCB7IFBhcnNlZFVybFF1ZXJ5SW5wdXQgfSBmcm9tICdxdWVyeXN0cmluZyc7XG5pbXBvcnQgeyByZW1vdmVGaXJzdFNsYXNoIH0gZnJvbSAnLi4vLi4vY29tcG9uZW50cy93b3Jrc3BhY2VzL3V0aWxzJztcbmltcG9ydCB7XG4gIGZ1bmN0aW9uc19jb25maWcsXG4gIGdldF9mb2xkZXJzX2FuZF9wbG90c19vbGRfYXBpLFxuICBnZXRfZm9sZGVyc19hbmRfcGxvdHNfbmV3X2FwaSxcbiAgZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX25ld19hcGlfd2l0aF9saXZlX21vZGUsXG4gIGdldF9ydW5fbGlzdF9ieV9zZWFyY2hfbmV3X2FwaSxcbiAgZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9vbGRfYXBpLFxuICBnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX25ld19hcGlfd2l0aF9ub19vbGRlcl90aGFuLFxufSBmcm9tICcuLi8uLi9jb25maWcvY29uZmlnJztcblxuZXhwb3J0IGNvbnN0IGdldEZvbGRlclBhdGggPSAoZm9sZGVyczogc3RyaW5nW10sIGNsaWNrZWRGb2xkZXI6IHN0cmluZykgPT4ge1xuICBjb25zdCBmb2xkZXJJbmRleCA9IGZvbGRlcnMuaW5kZXhPZihjbGlja2VkRm9sZGVyKTtcbiAgY29uc3QgcmVzdEZvbGRlcnM6IHN0cmluZ1tdID0gZm9sZGVycy5zbGljZSgwLCBmb2xkZXJJbmRleCArIDEpO1xuICBjb25zdCBmb2xkZXJzU3RyaW5nID0gcmVzdEZvbGRlcnMuam9pbignLycpO1xuICByZXR1cm4gZm9sZGVyc1N0cmluZztcbn07XG5cbmV4cG9ydCBjb25zdCBpc1Bsb3RTZWxlY3RlZCA9IChcbiAgc2VsZWN0ZWRfcGxvdHM6IFBsb3REYXRhUHJvcHNbXSxcbiAgcGxvdF9uYW1lOiBzdHJpbmdcbikgPT5cbiAgc2VsZWN0ZWRfcGxvdHMuc29tZShcbiAgICAoc2VsZWN0ZWRfcGxvdDogUGxvdERhdGFQcm9wcykgPT4gc2VsZWN0ZWRfcGxvdC5uYW1lID09PSBwbG90X25hbWVcbiAgKTtcblxuZXhwb3J0IGNvbnN0IGdldFNlbGVjdGVkUGxvdHNOYW1lcyA9IChwbG90c05hbWVzOiBzdHJpbmcgfCB1bmRlZmluZWQpID0+IHtcbiAgY29uc3QgcGxvdHMgPSBwbG90c05hbWVzID8gcGxvdHNOYW1lcy5zcGxpdCgnLycpIDogW107XG5cbiAgcmV0dXJuIHBsb3RzO1xufTtcblxuZXhwb3J0IGNvbnN0IGdldFNlbGVjdGVkUGxvdHMgPSAoXG4gIHBsb3RzUXVlcnk6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgcGxvdHM6IFBsb3REYXRhUHJvcHNbXVxuKSA9PiB7XG4gIGNvbnN0IHBsb3RzV2l0aERpcnMgPSBwbG90c1F1ZXJ5ID8gcGxvdHNRdWVyeS5zcGxpdCgnJicpIDogW107XG4gIHJldHVybiBwbG90c1dpdGhEaXJzLm1hcCgocGxvdFdpdGhEaXI6IHN0cmluZykgPT4ge1xuICAgIGNvbnN0IHBsb3RBbmREaXIgPSBwbG90V2l0aERpci5zcGxpdCgnLycpO1xuICAgIGNvbnN0IG5hbWUgPSBwbG90QW5kRGlyLnBvcCgpO1xuICAgIGNvbnN0IGRpcmVjdG9yaWVzID0gcGxvdEFuZERpci5qb2luKCcvJyk7XG4gICAgY29uc3QgcGxvdCA9IHBsb3RzLmZpbHRlcihcbiAgICAgIChwbG90KSA9PiBwbG90Lm5hbWUgPT09IG5hbWUgJiYgcGxvdC5wYXRoID09PSBkaXJlY3Rvcmllc1xuICAgICk7XG4gICAgY29uc3QgZGlzcGxheWVkTmFtZSA9XG4gICAgICBwbG90Lmxlbmd0aCA+IDAgJiYgcGxvdFswXS5kaXNwbGF5ZWROYW1lID8gcGxvdFswXS5kaXNwbGF5ZWROYW1lIDogJyc7XG5cbiAgICBjb25zdCBxcmVzdWx0cyA9IHBsb3RbMF0gJiYgcGxvdFswXS5xcmVzdWx0cztcblxuICAgIGNvbnN0IHBsb3RPYmplY3Q6IFBsb3REYXRhUHJvcHMgPSB7XG4gICAgICBuYW1lOiBuYW1lID8gbmFtZSA6ICcnLFxuICAgICAgcGF0aDogZGlyZWN0b3JpZXMsXG4gICAgICBkaXNwbGF5ZWROYW1lOiBkaXNwbGF5ZWROYW1lLFxuICAgICAgcXJlc3VsdHM6IHFyZXN1bHRzLFxuICAgIH07XG4gICAgcmV0dXJuIHBsb3RPYmplY3Q7XG4gIH0pO1xufTtcblxuZXhwb3J0IGNvbnN0IGdldEZvbGRlclBhdGhUb1F1ZXJ5ID0gKFxuICBwcmV2aXVvc0ZvbGRlclBhdGg6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgY3VycmVudFNlbGVjdGVkOiBzdHJpbmdcbikgPT4ge1xuICByZXR1cm4gcHJldml1b3NGb2xkZXJQYXRoXG4gICAgPyBgJHtwcmV2aXVvc0ZvbGRlclBhdGh9LyR7Y3VycmVudFNlbGVjdGVkfWBcbiAgICA6IGAvJHtjdXJyZW50U2VsZWN0ZWR9YDtcbn07XG5cbi8vIHdoYXQgaXMgc3RyZWFtZXJpbmZvPyAoY29taW5nIGZyb20gYXBpLCB3ZSBkb24ndCBrbm93IHdoYXQgaXQgaXMsIHNvIHdlIGZpbHRlcmVkIGl0IG91dClcbi8vIGdldENvbnRlbnQgYWxzbyBzb3J0aW5nIGRhdGEgdGhhdCBkaXJlY3RvcmllcyBzaG91bGQgYmUgZGlzcGxheWVkIGZpcnN0bHksIGp1c3QgYWZ0ZXIgdGhlbS0gcGxvdHMgaW1hZ2VzLlxuZXhwb3J0IGNvbnN0IGdldENvbnRlbnRzID0gKGRhdGE6IGFueSkgPT4ge1xuICBpZiAoZnVuY3Rpb25zX2NvbmZpZy5uZXdfYmFja19lbmQubmV3X2JhY2tfZW5kKSB7XG4gICAgcmV0dXJuIGRhdGEgPyBfLnNvcnRCeShkYXRhLmRhdGEgPyBkYXRhLmRhdGEgOiBbXSwgWydzdWJkaXInXSkgOiBbXTtcbiAgfVxuICByZXR1cm4gZGF0YVxuICAgID8gXy5zb3J0QnkoXG4gICAgICAgIGRhdGEuY29udGVudHNcbiAgICAgICAgICA/IGRhdGEuY29udGVudHNcbiAgICAgICAgICA6IFtdLmZpbHRlcihcbiAgICAgICAgICAgICAgKG9uZV9pdGVtOiBQbG90SW50ZXJmYWNlIHwgRGlyZWN0b3J5SW50ZXJmYWNlKSA9PlxuICAgICAgICAgICAgICAgICFvbmVfaXRlbS5oYXNPd25Qcm9wZXJ0eSgnc3RyZWFtZXJpbmZvJylcbiAgICAgICAgICAgICksXG4gICAgICAgIFsnc3ViZGlyJ11cbiAgICAgIClcbiAgICA6IFtdO1xufTtcblxuZXhwb3J0IGNvbnN0IGdldERpcmVjdG9yaWVzOiBhbnkgPSAoY29udGVudHM6IERpcmVjdG9yeUludGVyZmFjZVtdKSA9PiB7XG4gIHJldHVybiBjbGVhbkRlZXAoXG4gICAgY29udGVudHMubWFwKChjb250ZW50OiBEaXJlY3RvcnlJbnRlcmZhY2UpID0+IHtcbiAgICAgIGlmIChmdW5jdGlvbnNfY29uZmlnLm5ld19iYWNrX2VuZC5uZXdfYmFja19lbmQpIHtcbiAgICAgICAgcmV0dXJuIHsgc3ViZGlyOiBjb250ZW50LnN1YmRpciwgbWVfY291bnQ6IGNvbnRlbnQubWVfY291bnQgfTtcbiAgICAgIH1cbiAgICAgIHJldHVybiB7IHN1YmRpcjogY29udGVudC5zdWJkaXIgfTtcbiAgICB9KVxuICApO1xufTtcblxuZXhwb3J0IGNvbnN0IGdldEZvcm1hdGVkUGxvdHNPYmplY3QgPSAoY29udGVudHM6IFBsb3RJbnRlcmZhY2VbXSkgPT5cbiAgY2xlYW5EZWVwKFxuICAgIGNvbnRlbnRzLm1hcCgoY29udGVudDogUGxvdEludGVyZmFjZSkgPT4ge1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgZGlzcGxheWVkTmFtZTogY29udGVudC5vYmosXG4gICAgICAgIHBhdGg6IGNvbnRlbnQucGF0aCAmJiAnLycgKyBjb250ZW50LnBhdGgsXG4gICAgICAgIHByb3BlcnRpZXM6IGNvbnRlbnQucHJvcGVydGllcyxcbiAgICAgIH07XG4gICAgfSlcbiAgKS5zb3J0KCk7XG5cbmV4cG9ydCBjb25zdCBnZXRGaWx0ZXJlZERpcmVjdG9yaWVzID0gKFxuICBwbG90X3NlYXJjaF9mb2xkZXJzOiBEaXJlY3RvcnlJbnRlcmZhY2VbXSxcbiAgd29ya3NwYWNlX2ZvbGRlcnM6IChEaXJlY3RvcnlJbnRlcmZhY2UgfCB1bmRlZmluZWQpW11cbikgPT4ge1xuICAvL2lmIHdvcmtzcGFjZUZvbGRlcnMgYXJyYXkgZnJvbSBjb250ZXh0IGlzIG5vdCBlbXB0eSB3ZSB0YWtpbmcgaW50ZXJzZWN0aW9uIGJldHdlZW4gYWxsIGRpcmVjdG9yaWVzIGFuZCB3b3Jrc3BhY2VGb2xkZXJzXG4gIC8vIHdvcmtzcGFjZSBmb2xkZXJzIGFyZSBmaWxldGVyZCBmb2xkZXJzIGFycmF5IGJ5IHNlbGVjdGVkIHdvcmtzcGFjZVxuICBpZiAod29ya3NwYWNlX2ZvbGRlcnMubGVuZ3RoID4gMCkge1xuICAgIGNvbnN0IG5hbWVzX29mX2ZvbGRlcnMgPSBwbG90X3NlYXJjaF9mb2xkZXJzLm1hcChcbiAgICAgIChmb2xkZXI6IERpcmVjdG9yeUludGVyZmFjZSkgPT4gZm9sZGVyLnN1YmRpclxuICAgICk7XG4gICAgLy9AdHMtaWdub3JlXG4gICAgY29uc3QgZmlsdGVyZWREaXJlY3RvcmllcyA9IHdvcmtzcGFjZV9mb2xkZXJzLmZpbHRlcihcbiAgICAgIChkaXJlY3Rvcnk6IERpcmVjdG9yeUludGVyZmFjZSB8IHVuZGVmaW5lZCkgPT5cbiAgICAgICAgZGlyZWN0b3J5ICYmIG5hbWVzX29mX2ZvbGRlcnMuaW5jbHVkZXMoZGlyZWN0b3J5LnN1YmRpcilcbiAgICApO1xuICAgIHJldHVybiBmaWx0ZXJlZERpcmVjdG9yaWVzO1xuICB9XG4gIC8vIGlmIGZvbGRlcl9wYXRoIGFuZCB3b3Jrc3BhY2VGb2xkZXJzIGFyZSBlbXB0eSwgd2UgcmV0dXJuIGFsbCBkaXJlc3Rvcmllc1xuICBlbHNlIGlmICh3b3Jrc3BhY2VfZm9sZGVycy5sZW5ndGggPT09IDApIHtcbiAgICByZXR1cm4gcGxvdF9zZWFyY2hfZm9sZGVycztcbiAgfVxufTtcblxuZXhwb3J0IGNvbnN0IGdldENoYW5nZWRRdWVyeVBhcmFtcyA9IChcbiAgcGFyYW1zOiBQYXJzZWRVcmxRdWVyeUlucHV0LFxuICBxdWVyeTogUXVlcnlQcm9wc1xuKSA9PiB7XG4gIHBhcmFtcy5kYXRhc2V0X25hbWUgPSBwYXJhbXMuZGF0YXNldF9uYW1lXG4gICAgPyBwYXJhbXMuZGF0YXNldF9uYW1lXG4gICAgOiBkZWNvZGVVUklDb21wb25lbnQocXVlcnkuZGF0YXNldF9uYW1lIGFzIHN0cmluZyk7XG5cbiAgcGFyYW1zLnJ1bl9udW1iZXIgPSBwYXJhbXMucnVuX251bWJlciA/IHBhcmFtcy5ydW5fbnVtYmVyIDogcXVlcnkucnVuX251bWJlcjtcblxuICBwYXJhbXMuZm9sZGVyX3BhdGggPSBwYXJhbXMuZm9sZGVyX3BhdGhcbiAgICA/IHJlbW92ZUZpcnN0U2xhc2gocGFyYW1zLmZvbGRlcl9wYXRoIGFzIHN0cmluZylcbiAgICA6IHF1ZXJ5LmZvbGRlcl9wYXRoO1xuXG4gIHBhcmFtcy53b3Jrc3BhY2VzID0gcGFyYW1zLndvcmtzcGFjZXMgPyBwYXJhbXMud29ya3NwYWNlcyA6IHF1ZXJ5LndvcmtzcGFjZXM7XG5cbiAgcGFyYW1zLm92ZXJsYXkgPSBwYXJhbXMub3ZlcmxheSA/IHBhcmFtcy5vdmVybGF5IDogcXVlcnkub3ZlcmxheTtcblxuICBwYXJhbXMub3ZlcmxheV9kYXRhID1cbiAgICBwYXJhbXMub3ZlcmxheV9kYXRhID09PSAnJyB8fCBwYXJhbXMub3ZlcmxheV9kYXRhXG4gICAgICA/IHBhcmFtcy5vdmVybGF5X2RhdGFcbiAgICAgIDogcXVlcnkub3ZlcmxheV9kYXRhO1xuXG4gIHBhcmFtcy5zZWxlY3RlZF9wbG90cyA9XG4gICAgcGFyYW1zLnNlbGVjdGVkX3Bsb3RzID09PSAnJyB8fCBwYXJhbXMuc2VsZWN0ZWRfcGxvdHNcbiAgICAgID8gcGFyYW1zLnNlbGVjdGVkX3Bsb3RzXG4gICAgICA6IHF1ZXJ5LnNlbGVjdGVkX3Bsb3RzO1xuXG4gIC8vIGlmIHZhbHVlIG9mIHNlYXJjaCBmaWVsZCBpcyBlbXB0eSBzdHJpbmcsIHNob3VsZCBiZSByZXR1bmVkIGFsbCBmb2xkZXJzLlxuICAvLyBpZiBwYXJhbXMucGxvdF9zZWFyY2ggPT0gJycgd2hlbiByZXF1ZXN0IGlzIGRvbmUsIHBhcmFtcy5wbG90X3NlYXJjaCBpcyBjaGFuZ2VkIHRvIC4qXG4gIHBhcmFtcy5wbG90X3NlYXJjaCA9XG4gICAgcGFyYW1zLnBsb3Rfc2VhcmNoID09PSAnJyB8fCBwYXJhbXMucGxvdF9zZWFyY2hcbiAgICAgID8gcGFyYW1zLnBsb3Rfc2VhcmNoXG4gICAgICA6IHF1ZXJ5LnBsb3Rfc2VhcmNoO1xuXG4gIHBhcmFtcy5vdmVybGF5ID0gcGFyYW1zLm92ZXJsYXkgPyBwYXJhbXMub3ZlcmxheSA6IHF1ZXJ5Lm92ZXJsYXk7XG5cbiAgcGFyYW1zLm5vcm1hbGl6ZSA9IHBhcmFtcy5ub3JtYWxpemUgPyBwYXJhbXMubm9ybWFsaXplIDogcXVlcnkubm9ybWFsaXplO1xuXG4gIHBhcmFtcy5sdW1pID0gcGFyYW1zLmx1bWkgfHwgcGFyYW1zLmx1bWkgPT09IDAgPyBwYXJhbXMubHVtaSA6IHF1ZXJ5Lmx1bWk7XG5cbiAgLy9jbGVhbmluZyB1cmw6IGlmIHdvcmtzcGFjZSBpcyBub3Qgc2V0IChpdCBtZWFucyBpdCdzIGVtcHR5IHN0cmluZyksIGl0IHNob3VsZG4ndCBiZSB2aXNpYmxlIGluIHVybFxuICBjb25zdCBjbGVhbmVkX3BhcmFtZXRlcnMgPSBjbGVhbkRlZXAocGFyYW1zKTtcblxuICByZXR1cm4gY2xlYW5lZF9wYXJhbWV0ZXJzO1xufTtcblxuZXhwb3J0IGNvbnN0IGNoYW5nZVJvdXRlciA9IChwYXJhbWV0ZXJzOiBQYXJzZWRVcmxRdWVyeUlucHV0KSA9PiB7XG4gIGNvbnN0IHF1ZXJ5U3RyaW5nID0gcXMuc3RyaW5naWZ5KHBhcmFtZXRlcnMsIHt9KTtcbiAgUm91dGVyLnB1c2goe1xuICAgIHBhdGhuYW1lOiBgL2AsXG4gICAgcXVlcnk6IHBhcmFtZXRlcnMsXG4gICAgcGF0aDogZGVjb2RlVVJJQ29tcG9uZW50KHF1ZXJ5U3RyaW5nKSxcbiAgfSk7XG59O1xuXG5leHBvcnQgY29uc3QgZ2V0TmFtZUFuZERpcmVjdG9yaWVzRnJvbURpciA9IChjb250ZW50OiBQbG90SW50ZXJmYWNlKSA9PiB7XG4gIGNvbnN0IGRpciA9IGNvbnRlbnQucGF0aDtcbiAgY29uc3QgcGFydHNPZkRpciA9IGRpci5zcGxpdCgnLycpO1xuICBjb25zdCBuYW1lID0gcGFydHNPZkRpci5wb3AoKTtcbiAgY29uc3QgZGlyZWN0b3JpZXMgPSBwYXJ0c09mRGlyLmpvaW4oJy8nKTtcblxuICByZXR1cm4geyBuYW1lLCBkaXJlY3RvcmllcyB9O1xufTtcblxuZXhwb3J0IGNvbnN0IGlzX3J1bl9zZWxlY3RlZF9hbHJlYWR5ID0gKFxuICBydW46IHsgcnVuX251bWJlcjogc3RyaW5nOyBkYXRhc2V0X25hbWU6IHN0cmluZyB9LFxuICBxdWVyeTogUXVlcnlQcm9wc1xuKSA9PiB7XG4gIHJldHVybiAoXG4gICAgcnVuLnJ1bl9udW1iZXIgPT09IHF1ZXJ5LnJ1bl9udW1iZXIgJiZcbiAgICBydW4uZGF0YXNldF9uYW1lID09PSBxdWVyeS5kYXRhc2V0X25hbWVcbiAgKTtcbn07XG5cbmV4cG9ydCBjb25zdCBjaG9vc2VfYXBpID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMpID0+IHtcbiAgY29uc3QgY3VycmVudF9hcGkgPSAhZnVuY3Rpb25zX2NvbmZpZy5uZXdfYmFja19lbmQubmV3X2JhY2tfZW5kXG4gICAgPyBnZXRfZm9sZGVyc19hbmRfcGxvdHNfb2xkX2FwaShwYXJhbXMpXG4gICAgOiBmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnXG4gICAgPyBnZXRfZm9sZGVyc19hbmRfcGxvdHNfbmV3X2FwaV93aXRoX2xpdmVfbW9kZShwYXJhbXMpXG4gICAgOiBnZXRfZm9sZGVyc19hbmRfcGxvdHNfbmV3X2FwaShwYXJhbXMpO1xuICByZXR1cm4gY3VycmVudF9hcGk7XG59O1xuXG5leHBvcnQgY29uc3QgY2hvb3NlX2FwaV9mb3JfcnVuX3NlYXJjaCA9IChwYXJhbXM6IFBhcmFtc0ZvckFwaVByb3BzKSA9PiB7XG4gIGNvbnN0IGN1cnJlbnRfYXBpID0gIWZ1bmN0aW9uc19jb25maWcubmV3X2JhY2tfZW5kLm5ld19iYWNrX2VuZFxuICAgID8gZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9vbGRfYXBpKHBhcmFtcylcbiAgICA6IGZ1bmN0aW9uc19jb25maWcubW9kZSA9PT0gJ09OTElORSdcbiAgICA/IGdldF9ydW5fbGlzdF9ieV9zZWFyY2hfbmV3X2FwaV93aXRoX25vX29sZGVyX3RoYW4ocGFyYW1zKVxuICAgIDogZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9uZXdfYXBpKHBhcmFtcyk7XG5cbiAgcmV0dXJuIGN1cnJlbnRfYXBpO1xufTtcbiJdLCJzb3VyY2VSb290IjoiIn0=