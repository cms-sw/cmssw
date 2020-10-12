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
/* harmony import */ var _components_utils__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../components/utils */ "./components/utils.ts");







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
  params.workspace = params.workspace ? params.workspace : query.workspaces;
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
    pathname: Object(_components_utils__WEBPACK_IMPORTED_MODULE_6__["getPathName"])(),
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGFpbmVycy9kaXNwbGF5L3V0aWxzLnRzIl0sIm5hbWVzIjpbImdldEZvbGRlclBhdGgiLCJmb2xkZXJzIiwiY2xpY2tlZEZvbGRlciIsImZvbGRlckluZGV4IiwiaW5kZXhPZiIsInJlc3RGb2xkZXJzIiwic2xpY2UiLCJmb2xkZXJzU3RyaW5nIiwiam9pbiIsImlzUGxvdFNlbGVjdGVkIiwic2VsZWN0ZWRfcGxvdHMiLCJwbG90X25hbWUiLCJzb21lIiwic2VsZWN0ZWRfcGxvdCIsIm5hbWUiLCJnZXRTZWxlY3RlZFBsb3RzTmFtZXMiLCJwbG90c05hbWVzIiwicGxvdHMiLCJzcGxpdCIsImdldFNlbGVjdGVkUGxvdHMiLCJwbG90c1F1ZXJ5IiwicGxvdHNXaXRoRGlycyIsIm1hcCIsInBsb3RXaXRoRGlyIiwicGxvdEFuZERpciIsInBvcCIsImRpcmVjdG9yaWVzIiwicGxvdCIsImZpbHRlciIsInBhdGgiLCJkaXNwbGF5ZWROYW1lIiwibGVuZ3RoIiwicXJlc3VsdHMiLCJwbG90T2JqZWN0IiwiZ2V0Rm9sZGVyUGF0aFRvUXVlcnkiLCJwcmV2aXVvc0ZvbGRlclBhdGgiLCJjdXJyZW50U2VsZWN0ZWQiLCJnZXRDb250ZW50cyIsImRhdGEiLCJmdW5jdGlvbnNfY29uZmlnIiwibmV3X2JhY2tfZW5kIiwiXyIsInNvcnRCeSIsImNvbnRlbnRzIiwib25lX2l0ZW0iLCJoYXNPd25Qcm9wZXJ0eSIsImdldERpcmVjdG9yaWVzIiwiY2xlYW5EZWVwIiwiY29udGVudCIsInN1YmRpciIsIm1lX2NvdW50IiwiZ2V0Rm9ybWF0ZWRQbG90c09iamVjdCIsIm9iaiIsInByb3BlcnRpZXMiLCJzb3J0IiwiZ2V0RmlsdGVyZWREaXJlY3RvcmllcyIsInBsb3Rfc2VhcmNoX2ZvbGRlcnMiLCJ3b3Jrc3BhY2VfZm9sZGVycyIsIm5hbWVzX29mX2ZvbGRlcnMiLCJmb2xkZXIiLCJmaWx0ZXJlZERpcmVjdG9yaWVzIiwiZGlyZWN0b3J5IiwiaW5jbHVkZXMiLCJnZXRDaGFuZ2VkUXVlcnlQYXJhbXMiLCJwYXJhbXMiLCJxdWVyeSIsImRhdGFzZXRfbmFtZSIsImRlY29kZVVSSUNvbXBvbmVudCIsInJ1bl9udW1iZXIiLCJmb2xkZXJfcGF0aCIsInJlbW92ZUZpcnN0U2xhc2giLCJ3b3Jrc3BhY2UiLCJ3b3Jrc3BhY2VzIiwib3ZlcmxheSIsIm92ZXJsYXlfZGF0YSIsInBsb3Rfc2VhcmNoIiwibm9ybWFsaXplIiwibHVtaSIsImNsZWFuZWRfcGFyYW1ldGVycyIsImNoYW5nZVJvdXRlciIsInBhcmFtZXRlcnMiLCJxdWVyeVN0cmluZyIsInFzIiwic3RyaW5naWZ5IiwiUm91dGVyIiwicHVzaCIsInBhdGhuYW1lIiwiZ2V0UGF0aE5hbWUiLCJnZXROYW1lQW5kRGlyZWN0b3JpZXNGcm9tRGlyIiwiZGlyIiwicGFydHNPZkRpciIsImlzX3J1bl9zZWxlY3RlZF9hbHJlYWR5IiwicnVuIiwiY2hvb3NlX2FwaSIsImN1cnJlbnRfYXBpIiwiZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX29sZF9hcGkiLCJtb2RlIiwiZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX25ld19hcGlfd2l0aF9saXZlX21vZGUiLCJnZXRfZm9sZGVyc19hbmRfcGxvdHNfbmV3X2FwaSIsImNob29zZV9hcGlfZm9yX3J1bl9zZWFyY2giLCJnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX29sZF9hcGkiLCJnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX25ld19hcGlfd2l0aF9ub19vbGRlcl90aGFuIiwiZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9uZXdfYXBpIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7O0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFDQTtBQUNBO0FBU0E7QUFFQTtBQUNB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Q0F3QixHQUFHLENBQUMxQixZQUFKLEtBQXFCRCxLQUFLLENBQUNDLFlBRjdCO0FBSUQsQ0FSTTtBQVVBLElBQU0yQixVQUFVLEdBQUcsU0FBYkEsVUFBYSxDQUFDN0IsTUFBRCxFQUErQjtBQUN2RCxNQUFNOEIsV0FBVyxHQUFHLENBQUN2RCwrREFBZ0IsQ0FBQ0MsWUFBakIsQ0FBOEJBLFlBQS9CLEdBQ2hCdUQsb0ZBQTZCLENBQUMvQixNQUFELENBRGIsR0FFaEJ6QiwrREFBZ0IsQ0FBQ3lELElBQWpCLEtBQTBCLFFBQTFCLEdBQ0FDLG1HQUE0QyxDQUFDakMsTUFBRCxDQUQ1QyxHQUVBa0Msb0ZBQTZCLENBQUNsQyxNQUFELENBSmpDO0FBS0EsU0FBTzhCLFdBQVA7QUFDRCxDQVBNO0FBU0EsSUFBTUsseUJBQXlCLEdBQUcsU0FBNUJBLHlCQUE0QixDQUFDbkMsTUFBRCxFQUErQjtBQUN0RSxNQUFNOEIsV0FBVyxHQUFHLENBQUN2RCwrREFBZ0IsQ0FBQ0MsWUFBakIsQ0FBOEJBLFlBQS9CLEdBQ2hCNEQscUZBQThCLENBQUNwQyxNQUFELENBRGQsR0FFaEJ6QiwrREFBZ0IsQ0FBQ3lELElBQWpCLEtBQTBCLFFBQTFCLEdBQ0FLLHdHQUFpRCxDQUFDckMsTUFBRCxDQURqRCxHQUVBc0MscUZBQThCLENBQUN0QyxNQUFELENBSmxDO0FBTUEsU0FBTzhCLFdBQVA7QUFDRCxDQVJNIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmJjOGYzOTRlY2M5YzM0MzY1MzIzLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgY2xlYW5EZWVwIGZyb20gJ2NsZWFuLWRlZXAnO1xuaW1wb3J0IF8gZnJvbSAnbG9kYXNoJztcbmltcG9ydCBxcyBmcm9tICdxcyc7XG5cbmltcG9ydCB7XG4gIFBsb3REYXRhUHJvcHMsXG4gIFBsb3RJbnRlcmZhY2UsXG4gIERpcmVjdG9yeUludGVyZmFjZSxcbiAgUXVlcnlQcm9wcyxcbiAgUGFyYW1zRm9yQXBpUHJvcHMsXG59IGZyb20gJy4vaW50ZXJmYWNlcyc7XG5pbXBvcnQgUm91dGVyIGZyb20gJ25leHQvcm91dGVyJztcbmltcG9ydCB7IFBhcnNlZFVybFF1ZXJ5SW5wdXQgfSBmcm9tICdxdWVyeXN0cmluZyc7XG5pbXBvcnQgeyByZW1vdmVGaXJzdFNsYXNoIH0gZnJvbSAnLi4vLi4vY29tcG9uZW50cy93b3Jrc3BhY2VzL3V0aWxzJztcbmltcG9ydCB7XG4gIGZ1bmN0aW9uc19jb25maWcsXG4gIGdldF9mb2xkZXJzX2FuZF9wbG90c19vbGRfYXBpLFxuICBnZXRfZm9sZGVyc19hbmRfcGxvdHNfbmV3X2FwaSxcbiAgZ2V0X2ZvbGRlcnNfYW5kX3Bsb3RzX25ld19hcGlfd2l0aF9saXZlX21vZGUsXG4gIGdldF9ydW5fbGlzdF9ieV9zZWFyY2hfbmV3X2FwaSxcbiAgZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9vbGRfYXBpLFxuICBnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX25ld19hcGlfd2l0aF9ub19vbGRlcl90aGFuLFxufSBmcm9tICcuLi8uLi9jb25maWcvY29uZmlnJztcbmltcG9ydCB7IGdldFBhdGhOYW1lIH0gZnJvbSAnLi4vLi4vY29tcG9uZW50cy91dGlscyc7XG5cbmV4cG9ydCBjb25zdCBnZXRGb2xkZXJQYXRoID0gKGZvbGRlcnM6IHN0cmluZ1tdLCBjbGlja2VkRm9sZGVyOiBzdHJpbmcpID0+IHtcbiAgY29uc3QgZm9sZGVySW5kZXggPSBmb2xkZXJzLmluZGV4T2YoY2xpY2tlZEZvbGRlcik7XG4gIGNvbnN0IHJlc3RGb2xkZXJzOiBzdHJpbmdbXSA9IGZvbGRlcnMuc2xpY2UoMCwgZm9sZGVySW5kZXggKyAxKTtcbiAgY29uc3QgZm9sZGVyc1N0cmluZyA9IHJlc3RGb2xkZXJzLmpvaW4oJy8nKTtcbiAgcmV0dXJuIGZvbGRlcnNTdHJpbmc7XG59O1xuXG5leHBvcnQgY29uc3QgaXNQbG90U2VsZWN0ZWQgPSAoXG4gIHNlbGVjdGVkX3Bsb3RzOiBQbG90RGF0YVByb3BzW10sXG4gIHBsb3RfbmFtZTogc3RyaW5nXG4pID0+XG4gIHNlbGVjdGVkX3Bsb3RzLnNvbWUoXG4gICAgKHNlbGVjdGVkX3Bsb3Q6IFBsb3REYXRhUHJvcHMpID0+IHNlbGVjdGVkX3Bsb3QubmFtZSA9PT0gcGxvdF9uYW1lXG4gICk7XG5cbmV4cG9ydCBjb25zdCBnZXRTZWxlY3RlZFBsb3RzTmFtZXMgPSAocGxvdHNOYW1lczogc3RyaW5nIHwgdW5kZWZpbmVkKSA9PiB7XG4gIGNvbnN0IHBsb3RzID0gcGxvdHNOYW1lcyA/IHBsb3RzTmFtZXMuc3BsaXQoJy8nKSA6IFtdO1xuXG4gIHJldHVybiBwbG90cztcbn07XG5cbmV4cG9ydCBjb25zdCBnZXRTZWxlY3RlZFBsb3RzID0gKFxuICBwbG90c1F1ZXJ5OiBzdHJpbmcgfCB1bmRlZmluZWQsXG4gIHBsb3RzOiBQbG90RGF0YVByb3BzW11cbikgPT4ge1xuICBjb25zdCBwbG90c1dpdGhEaXJzID0gcGxvdHNRdWVyeSA/IHBsb3RzUXVlcnkuc3BsaXQoJyYnKSA6IFtdO1xuICByZXR1cm4gcGxvdHNXaXRoRGlycy5tYXAoKHBsb3RXaXRoRGlyOiBzdHJpbmcpID0+IHtcbiAgICBjb25zdCBwbG90QW5kRGlyID0gcGxvdFdpdGhEaXIuc3BsaXQoJy8nKTtcbiAgICBjb25zdCBuYW1lID0gcGxvdEFuZERpci5wb3AoKTtcbiAgICBjb25zdCBkaXJlY3RvcmllcyA9IHBsb3RBbmREaXIuam9pbignLycpO1xuICAgIGNvbnN0IHBsb3QgPSBwbG90cy5maWx0ZXIoXG4gICAgICAocGxvdCkgPT4gcGxvdC5uYW1lID09PSBuYW1lICYmIHBsb3QucGF0aCA9PT0gZGlyZWN0b3JpZXNcbiAgICApO1xuICAgIGNvbnN0IGRpc3BsYXllZE5hbWUgPVxuICAgICAgcGxvdC5sZW5ndGggPiAwICYmIHBsb3RbMF0uZGlzcGxheWVkTmFtZSA/IHBsb3RbMF0uZGlzcGxheWVkTmFtZSA6ICcnO1xuXG4gICAgY29uc3QgcXJlc3VsdHMgPSBwbG90WzBdICYmIHBsb3RbMF0ucXJlc3VsdHM7XG5cbiAgICBjb25zdCBwbG90T2JqZWN0OiBQbG90RGF0YVByb3BzID0ge1xuICAgICAgbmFtZTogbmFtZSA/IG5hbWUgOiAnJyxcbiAgICAgIHBhdGg6IGRpcmVjdG9yaWVzLFxuICAgICAgZGlzcGxheWVkTmFtZTogZGlzcGxheWVkTmFtZSxcbiAgICAgIHFyZXN1bHRzOiBxcmVzdWx0cyxcbiAgICB9O1xuICAgIHJldHVybiBwbG90T2JqZWN0O1xuICB9KTtcbn07XG5cbmV4cG9ydCBjb25zdCBnZXRGb2xkZXJQYXRoVG9RdWVyeSA9IChcbiAgcHJldml1b3NGb2xkZXJQYXRoOiBzdHJpbmcgfCB1bmRlZmluZWQsXG4gIGN1cnJlbnRTZWxlY3RlZDogc3RyaW5nXG4pID0+IHtcbiAgcmV0dXJuIHByZXZpdW9zRm9sZGVyUGF0aFxuICAgID8gYCR7cHJldml1b3NGb2xkZXJQYXRofS8ke2N1cnJlbnRTZWxlY3RlZH1gXG4gICAgOiBgLyR7Y3VycmVudFNlbGVjdGVkfWA7XG59O1xuXG4vLyB3aGF0IGlzIHN0cmVhbWVyaW5mbz8gKGNvbWluZyBmcm9tIGFwaSwgd2UgZG9uJ3Qga25vdyB3aGF0IGl0IGlzLCBzbyB3ZSBmaWx0ZXJlZCBpdCBvdXQpXG4vLyBnZXRDb250ZW50IGFsc28gc29ydGluZyBkYXRhIHRoYXQgZGlyZWN0b3JpZXMgc2hvdWxkIGJlIGRpc3BsYXllZCBmaXJzdGx5LCBqdXN0IGFmdGVyIHRoZW0tIHBsb3RzIGltYWdlcy5cbmV4cG9ydCBjb25zdCBnZXRDb250ZW50cyA9IChkYXRhOiBhbnkpID0+IHtcbiAgaWYgKGZ1bmN0aW9uc19jb25maWcubmV3X2JhY2tfZW5kLm5ld19iYWNrX2VuZCkge1xuICAgIHJldHVybiBkYXRhID8gXy5zb3J0QnkoZGF0YS5kYXRhID8gZGF0YS5kYXRhIDogW10sIFsnc3ViZGlyJ10pIDogW107XG4gIH1cbiAgcmV0dXJuIGRhdGFcbiAgICA/IF8uc29ydEJ5KFxuICAgICAgICBkYXRhLmNvbnRlbnRzXG4gICAgICAgICAgPyBkYXRhLmNvbnRlbnRzXG4gICAgICAgICAgOiBbXS5maWx0ZXIoXG4gICAgICAgICAgICAgIChvbmVfaXRlbTogUGxvdEludGVyZmFjZSB8IERpcmVjdG9yeUludGVyZmFjZSkgPT5cbiAgICAgICAgICAgICAgICAhb25lX2l0ZW0uaGFzT3duUHJvcGVydHkoJ3N0cmVhbWVyaW5mbycpXG4gICAgICAgICAgICApLFxuICAgICAgICBbJ3N1YmRpciddXG4gICAgICApXG4gICAgOiBbXTtcbn07XG5cbmV4cG9ydCBjb25zdCBnZXREaXJlY3RvcmllczogYW55ID0gKGNvbnRlbnRzOiBEaXJlY3RvcnlJbnRlcmZhY2VbXSkgPT4ge1xuICByZXR1cm4gY2xlYW5EZWVwKFxuICAgIGNvbnRlbnRzLm1hcCgoY29udGVudDogRGlyZWN0b3J5SW50ZXJmYWNlKSA9PiB7XG4gICAgICBpZiAoZnVuY3Rpb25zX2NvbmZpZy5uZXdfYmFja19lbmQubmV3X2JhY2tfZW5kKSB7XG4gICAgICAgIHJldHVybiB7IHN1YmRpcjogY29udGVudC5zdWJkaXIsIG1lX2NvdW50OiBjb250ZW50Lm1lX2NvdW50IH07XG4gICAgICB9XG4gICAgICByZXR1cm4geyBzdWJkaXI6IGNvbnRlbnQuc3ViZGlyIH07XG4gICAgfSlcbiAgKTtcbn07XG5cbmV4cG9ydCBjb25zdCBnZXRGb3JtYXRlZFBsb3RzT2JqZWN0ID0gKGNvbnRlbnRzOiBQbG90SW50ZXJmYWNlW10pID0+XG4gIGNsZWFuRGVlcChcbiAgICBjb250ZW50cy5tYXAoKGNvbnRlbnQ6IFBsb3RJbnRlcmZhY2UpID0+IHtcbiAgICAgIHJldHVybiB7XG4gICAgICAgIGRpc3BsYXllZE5hbWU6IGNvbnRlbnQub2JqLFxuICAgICAgICBwYXRoOiBjb250ZW50LnBhdGggJiYgJy8nICsgY29udGVudC5wYXRoLFxuICAgICAgICBwcm9wZXJ0aWVzOiBjb250ZW50LnByb3BlcnRpZXMsXG4gICAgICB9O1xuICAgIH0pXG4gICkuc29ydCgpO1xuXG5leHBvcnQgY29uc3QgZ2V0RmlsdGVyZWREaXJlY3RvcmllcyA9IChcbiAgcGxvdF9zZWFyY2hfZm9sZGVyczogRGlyZWN0b3J5SW50ZXJmYWNlW10sXG4gIHdvcmtzcGFjZV9mb2xkZXJzOiAoRGlyZWN0b3J5SW50ZXJmYWNlIHwgdW5kZWZpbmVkKVtdXG4pID0+IHtcbiAgLy9pZiB3b3Jrc3BhY2VGb2xkZXJzIGFycmF5IGZyb20gY29udGV4dCBpcyBub3QgZW1wdHkgd2UgdGFraW5nIGludGVyc2VjdGlvbiBiZXR3ZWVuIGFsbCBkaXJlY3RvcmllcyBhbmQgd29ya3NwYWNlRm9sZGVyc1xuICAvLyB3b3Jrc3BhY2UgZm9sZGVycyBhcmUgZmlsZXRlcmQgZm9sZGVycyBhcnJheSBieSBzZWxlY3RlZCB3b3Jrc3BhY2VcbiAgaWYgKHdvcmtzcGFjZV9mb2xkZXJzLmxlbmd0aCA+IDApIHtcbiAgICBjb25zdCBuYW1lc19vZl9mb2xkZXJzID0gcGxvdF9zZWFyY2hfZm9sZGVycy5tYXAoXG4gICAgICAoZm9sZGVyOiBEaXJlY3RvcnlJbnRlcmZhY2UpID0+IGZvbGRlci5zdWJkaXJcbiAgICApO1xuICAgIC8vQHRzLWlnbm9yZVxuICAgIGNvbnN0IGZpbHRlcmVkRGlyZWN0b3JpZXMgPSB3b3Jrc3BhY2VfZm9sZGVycy5maWx0ZXIoXG4gICAgICAoZGlyZWN0b3J5OiBEaXJlY3RvcnlJbnRlcmZhY2UgfCB1bmRlZmluZWQpID0+XG4gICAgICAgIGRpcmVjdG9yeSAmJiBuYW1lc19vZl9mb2xkZXJzLmluY2x1ZGVzKGRpcmVjdG9yeS5zdWJkaXIpXG4gICAgKTtcbiAgICByZXR1cm4gZmlsdGVyZWREaXJlY3RvcmllcztcbiAgfVxuICAvLyBpZiBmb2xkZXJfcGF0aCBhbmQgd29ya3NwYWNlRm9sZGVycyBhcmUgZW1wdHksIHdlIHJldHVybiBhbGwgZGlyZXN0b3JpZXNcbiAgZWxzZSBpZiAod29ya3NwYWNlX2ZvbGRlcnMubGVuZ3RoID09PSAwKSB7XG4gICAgcmV0dXJuIHBsb3Rfc2VhcmNoX2ZvbGRlcnM7XG4gIH1cbn07XG5cbmV4cG9ydCBjb25zdCBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMgPSAoXG4gIHBhcmFtczogUGFyc2VkVXJsUXVlcnlJbnB1dCxcbiAgcXVlcnk6IFF1ZXJ5UHJvcHNcbikgPT4ge1xuICBwYXJhbXMuZGF0YXNldF9uYW1lID0gcGFyYW1zLmRhdGFzZXRfbmFtZVxuICAgID8gcGFyYW1zLmRhdGFzZXRfbmFtZVxuICAgIDogZGVjb2RlVVJJQ29tcG9uZW50KHF1ZXJ5LmRhdGFzZXRfbmFtZSBhcyBzdHJpbmcpO1xuXG4gIHBhcmFtcy5ydW5fbnVtYmVyID0gcGFyYW1zLnJ1bl9udW1iZXIgPyBwYXJhbXMucnVuX251bWJlciA6IHF1ZXJ5LnJ1bl9udW1iZXI7XG5cbiAgcGFyYW1zLmZvbGRlcl9wYXRoID0gcGFyYW1zLmZvbGRlcl9wYXRoXG4gICAgPyByZW1vdmVGaXJzdFNsYXNoKHBhcmFtcy5mb2xkZXJfcGF0aCBhcyBzdHJpbmcpXG4gICAgOiBxdWVyeS5mb2xkZXJfcGF0aDtcblxuICBwYXJhbXMud29ya3NwYWNlID0gcGFyYW1zLndvcmtzcGFjZSA/IHBhcmFtcy53b3Jrc3BhY2UgOiBxdWVyeS53b3Jrc3BhY2VzO1xuXG4gIHBhcmFtcy5vdmVybGF5ID0gcGFyYW1zLm92ZXJsYXkgPyBwYXJhbXMub3ZlcmxheSA6IHF1ZXJ5Lm92ZXJsYXk7XG5cbiAgcGFyYW1zLm92ZXJsYXlfZGF0YSA9XG4gICAgcGFyYW1zLm92ZXJsYXlfZGF0YSA9PT0gJycgfHwgcGFyYW1zLm92ZXJsYXlfZGF0YVxuICAgICAgPyBwYXJhbXMub3ZlcmxheV9kYXRhXG4gICAgICA6IHF1ZXJ5Lm92ZXJsYXlfZGF0YTtcblxuICBwYXJhbXMuc2VsZWN0ZWRfcGxvdHMgPVxuICAgIHBhcmFtcy5zZWxlY3RlZF9wbG90cyA9PT0gJycgfHwgcGFyYW1zLnNlbGVjdGVkX3Bsb3RzXG4gICAgICA/IHBhcmFtcy5zZWxlY3RlZF9wbG90c1xuICAgICAgOiBxdWVyeS5zZWxlY3RlZF9wbG90cztcblxuICAvLyBpZiB2YWx1ZSBvZiBzZWFyY2ggZmllbGQgaXMgZW1wdHkgc3RyaW5nLCBzaG91bGQgYmUgcmV0dW5lZCBhbGwgZm9sZGVycy5cbiAgLy8gaWYgcGFyYW1zLnBsb3Rfc2VhcmNoID09ICcnIHdoZW4gcmVxdWVzdCBpcyBkb25lLCBwYXJhbXMucGxvdF9zZWFyY2ggaXMgY2hhbmdlZCB0byAuKlxuICBwYXJhbXMucGxvdF9zZWFyY2ggPVxuICAgIHBhcmFtcy5wbG90X3NlYXJjaCA9PT0gJycgfHwgcGFyYW1zLnBsb3Rfc2VhcmNoXG4gICAgICA/IHBhcmFtcy5wbG90X3NlYXJjaFxuICAgICAgOiBxdWVyeS5wbG90X3NlYXJjaDtcblxuICBwYXJhbXMub3ZlcmxheSA9IHBhcmFtcy5vdmVybGF5ID8gcGFyYW1zLm92ZXJsYXkgOiBxdWVyeS5vdmVybGF5O1xuXG4gIHBhcmFtcy5ub3JtYWxpemUgPSBwYXJhbXMubm9ybWFsaXplID8gcGFyYW1zLm5vcm1hbGl6ZSA6IHF1ZXJ5Lm5vcm1hbGl6ZTtcblxuICBwYXJhbXMubHVtaSA9IHBhcmFtcy5sdW1pIHx8IHBhcmFtcy5sdW1pID09PSAwID8gcGFyYW1zLmx1bWkgOiBxdWVyeS5sdW1pO1xuXG4gIC8vY2xlYW5pbmcgdXJsOiBpZiB3b3Jrc3BhY2UgaXMgbm90IHNldCAoaXQgbWVhbnMgaXQncyBlbXB0eSBzdHJpbmcpLCBpdCBzaG91bGRuJ3QgYmUgdmlzaWJsZSBpbiB1cmxcbiAgY29uc3QgY2xlYW5lZF9wYXJhbWV0ZXJzID0gY2xlYW5EZWVwKHBhcmFtcyk7XG5cbiAgcmV0dXJuIGNsZWFuZWRfcGFyYW1ldGVycztcbn07XG5cbmV4cG9ydCBjb25zdCBjaGFuZ2VSb3V0ZXIgPSAocGFyYW1ldGVyczogUGFyc2VkVXJsUXVlcnlJbnB1dCkgPT4ge1xuICBjb25zdCBxdWVyeVN0cmluZyA9IHFzLnN0cmluZ2lmeShwYXJhbWV0ZXJzLCB7fSk7XG4gIFJvdXRlci5wdXNoKHtcbiAgICBwYXRobmFtZTogZ2V0UGF0aE5hbWUoKSxcbiAgICBxdWVyeTogcGFyYW1ldGVycyxcbiAgICBwYXRoOiBkZWNvZGVVUklDb21wb25lbnQocXVlcnlTdHJpbmcpLFxuICB9KTtcbn07XG5cbmV4cG9ydCBjb25zdCBnZXROYW1lQW5kRGlyZWN0b3JpZXNGcm9tRGlyID0gKGNvbnRlbnQ6IFBsb3RJbnRlcmZhY2UpID0+IHtcbiAgY29uc3QgZGlyID0gY29udGVudC5wYXRoO1xuICBjb25zdCBwYXJ0c09mRGlyID0gZGlyLnNwbGl0KCcvJyk7XG4gIGNvbnN0IG5hbWUgPSBwYXJ0c09mRGlyLnBvcCgpO1xuICBjb25zdCBkaXJlY3RvcmllcyA9IHBhcnRzT2ZEaXIuam9pbignLycpO1xuXG4gIHJldHVybiB7IG5hbWUsIGRpcmVjdG9yaWVzIH07XG59O1xuXG5leHBvcnQgY29uc3QgaXNfcnVuX3NlbGVjdGVkX2FscmVhZHkgPSAoXG4gIHJ1bjogeyBydW5fbnVtYmVyOiBzdHJpbmc7IGRhdGFzZXRfbmFtZTogc3RyaW5nIH0sXG4gIHF1ZXJ5OiBRdWVyeVByb3BzXG4pID0+IHtcbiAgcmV0dXJuIChcbiAgICBydW4ucnVuX251bWJlciA9PT0gcXVlcnkucnVuX251bWJlciAmJlxuICAgIHJ1bi5kYXRhc2V0X25hbWUgPT09IHF1ZXJ5LmRhdGFzZXRfbmFtZVxuICApO1xufTtcblxuZXhwb3J0IGNvbnN0IGNob29zZV9hcGkgPSAocGFyYW1zOiBQYXJhbXNGb3JBcGlQcm9wcykgPT4ge1xuICBjb25zdCBjdXJyZW50X2FwaSA9ICFmdW5jdGlvbnNfY29uZmlnLm5ld19iYWNrX2VuZC5uZXdfYmFja19lbmRcbiAgICA/IGdldF9mb2xkZXJzX2FuZF9wbG90c19vbGRfYXBpKHBhcmFtcylcbiAgICA6IGZ1bmN0aW9uc19jb25maWcubW9kZSA9PT0gJ09OTElORSdcbiAgICA/IGdldF9mb2xkZXJzX2FuZF9wbG90c19uZXdfYXBpX3dpdGhfbGl2ZV9tb2RlKHBhcmFtcylcbiAgICA6IGdldF9mb2xkZXJzX2FuZF9wbG90c19uZXdfYXBpKHBhcmFtcyk7XG4gIHJldHVybiBjdXJyZW50X2FwaTtcbn07XG5cbmV4cG9ydCBjb25zdCBjaG9vc2VfYXBpX2Zvcl9ydW5fc2VhcmNoID0gKHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMpID0+IHtcbiAgY29uc3QgY3VycmVudF9hcGkgPSAhZnVuY3Rpb25zX2NvbmZpZy5uZXdfYmFja19lbmQubmV3X2JhY2tfZW5kXG4gICAgPyBnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX29sZF9hcGkocGFyYW1zKVxuICAgIDogZnVuY3Rpb25zX2NvbmZpZy5tb2RlID09PSAnT05MSU5FJ1xuICAgID8gZ2V0X3J1bl9saXN0X2J5X3NlYXJjaF9uZXdfYXBpX3dpdGhfbm9fb2xkZXJfdGhhbihwYXJhbXMpXG4gICAgOiBnZXRfcnVuX2xpc3RfYnlfc2VhcmNoX25ld19hcGkocGFyYW1zKTtcblxuICByZXR1cm4gY3VycmVudF9hcGk7XG59O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==