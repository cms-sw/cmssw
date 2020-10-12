webpackHotUpdate_N_E("pages/index",{

/***/ "./hooks/useFilterFoldersByWorkspace.tsx":
/*!***********************************************!*\
  !*** ./hooks/useFilterFoldersByWorkspace.tsx ***!
  \***********************************************/
/*! exports provided: useFilterFoldersByWorkspaces */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "useFilterFoldersByWorkspaces", function() { return useFilterFoldersByWorkspaces; });
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var lodash__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! lodash */ "./node_modules/lodash/lodash.js");
/* harmony import */ var lodash__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(lodash__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _components_workspaces_utils__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../components/workspaces/utils */ "./components/workspaces/utils.ts");
/* harmony import */ var _workspaces_offline__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../workspaces/offline */ "./workspaces/offline.ts");


var _s = $RefreshSig$();





var useFilterFoldersByWorkspaces = function useFilterFoldersByWorkspaces(query, watchers) {
  _s();

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_1__["useState"]([]),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState, 2),
      availableFolders = _React$useState2[0],
      setAvailableFolders = _React$useState2[1];

  var _React$useState3 = react__WEBPACK_IMPORTED_MODULE_1__["useState"]([]),
      _React$useState4 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState3, 2),
      filteredFolders = _React$useState4[0],
      setFilteredFolders = _React$useState4[1];

  var filteredInnerFolders = [];
  var workspace = s;
  var folderPathFromQuery = query.folder_path;
  var plot_search = query.plot_search;
  react__WEBPACK_IMPORTED_MODULE_1__["useEffect"](function () {
    //getting folderPath by selected workspace
    _workspaces_offline__WEBPACK_IMPORTED_MODULE_4__["workspaces"].forEach(function (workspaceFromList) {
      workspaceFromList.workspaces.forEach(function (oneWorkspace) {
        if (oneWorkspace.label === workspace) {
          setAvailableFolders(oneWorkspace.foldersPath);
        }
      });
    });
  }, [workspace]);
  react__WEBPACK_IMPORTED_MODULE_1__["useEffect"](function () {
    if (workspace && !folderPathFromQuery) {
      var firstLayerFolders = lodash__WEBPACK_IMPORTED_MODULE_2___default.a.uniq(availableFolders.map(function (foldersPath) {
        var firstLayer = foldersPath.split('/')[0];
        return firstLayer;
      }));

      var folders_object = firstLayerFolders.map(function (folder) {
        return {
          subdir: folder
        };
      });
      setFilteredFolders(folders_object);
    } else if (!!workspace && !!folderPathFromQuery) {
      availableFolders.forEach(function (foldersPath) {
        var folderPathFromQueryWithoutFirstSlash = Object(_components_workspaces_utils__WEBPACK_IMPORTED_MODULE_3__["removeFirstSlash"])(folderPathFromQuery); //if folderPath has a slash in the begining- removing it

        var matchBeginingInAvailableFolders = foldersPath.search(folderPathFromQueryWithoutFirstSlash); //searching in available folders, is clicked folder is part of availableFolders path

        if (matchBeginingInAvailableFolders >= 0) {
          // if selected folder is a part of available folderspath, we trying to get further layer folders.
          //matchEnd is the index, which indicates the end of seleced folder path string (we can see it in url) in available path
          // (availble path we set with setAvailableFolders action)
          var matchEnd = matchBeginingInAvailableFolders + folderPathFromQueryWithoutFirstSlash.length;
          var restFolders = foldersPath.substring(matchEnd, foldersPath.length);
          var firstLayerFolderOfRest = restFolders.split('/')[1]; //if it is the last layer firstLayerFolderOfRest will be undef.
          // in this case filteredInnerFolders will be empty array

          if (firstLayerFolderOfRest) {
            filteredInnerFolders.push(firstLayerFolderOfRest);
          }
        }
      });

      var _folders_object = filteredInnerFolders.map(function (folder) {
        //need to have the same format as directories got from api or by plot search
        return {
          subdir: folder
        };
      });

      setFilteredFolders(_folders_object);
    }
  }, [folderPathFromQuery, availableFolders, plot_search]);
  return {
    filteredFolders: filteredFolders
  };
};

_s(useFilterFoldersByWorkspaces, "Qsh0j1qYet55WdIDoo7gmYZ9/8U=");

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vaG9va3MvdXNlRmlsdGVyRm9sZGVyc0J5V29ya3NwYWNlLnRzeCJdLCJuYW1lcyI6WyJ1c2VGaWx0ZXJGb2xkZXJzQnlXb3Jrc3BhY2VzIiwicXVlcnkiLCJ3YXRjaGVycyIsIlJlYWN0IiwiYXZhaWxhYmxlRm9sZGVycyIsInNldEF2YWlsYWJsZUZvbGRlcnMiLCJmaWx0ZXJlZEZvbGRlcnMiLCJzZXRGaWx0ZXJlZEZvbGRlcnMiLCJmaWx0ZXJlZElubmVyRm9sZGVycyIsIndvcmtzcGFjZSIsInMiLCJmb2xkZXJQYXRoRnJvbVF1ZXJ5IiwiZm9sZGVyX3BhdGgiLCJwbG90X3NlYXJjaCIsIndvcmtzcGFjZXMiLCJmb3JFYWNoIiwid29ya3NwYWNlRnJvbUxpc3QiLCJvbmVXb3Jrc3BhY2UiLCJsYWJlbCIsImZvbGRlcnNQYXRoIiwiZmlyc3RMYXllckZvbGRlcnMiLCJfIiwidW5pcSIsIm1hcCIsImZpcnN0TGF5ZXIiLCJzcGxpdCIsImZvbGRlcnNfb2JqZWN0IiwiZm9sZGVyIiwic3ViZGlyIiwiZm9sZGVyUGF0aEZyb21RdWVyeVdpdGhvdXRGaXJzdFNsYXNoIiwicmVtb3ZlRmlyc3RTbGFzaCIsIm1hdGNoQmVnaW5pbmdJbkF2YWlsYWJsZUZvbGRlcnMiLCJzZWFyY2giLCJtYXRjaEVuZCIsImxlbmd0aCIsInJlc3RGb2xkZXJzIiwic3Vic3RyaW5nIiwiZmlyc3RMYXllckZvbGRlck9mUmVzdCIsInB1c2giXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQU1BO0FBQ0E7QUFHTyxJQUFNQSw0QkFBNEIsR0FBRyxTQUEvQkEsNEJBQStCLENBQzFDQyxLQUQwQyxFQUUxQ0MsUUFGMEMsRUFHdkM7QUFBQTs7QUFBQSx3QkFDNkNDLDhDQUFBLENBQXlCLEVBQXpCLENBRDdDO0FBQUE7QUFBQSxNQUNJQyxnQkFESjtBQUFBLE1BQ3NCQyxtQkFEdEI7O0FBQUEseUJBRTJDRiw4Q0FBQSxD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LCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNTZjNDU4MThlMmMyMWY3YjMwYmUuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCBfIGZyb20gJ2xvZGFzaCc7XG5cbmltcG9ydCB7XG4gIFF1ZXJ5UHJvcHMsXG4gIERpcmVjdG9yeUludGVyZmFjZSxcbn0gZnJvbSAnLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgcmVtb3ZlRmlyc3RTbGFzaCB9IGZyb20gJy4uL2NvbXBvbmVudHMvd29ya3NwYWNlcy91dGlscyc7XG5pbXBvcnQgeyB3b3Jrc3BhY2VzIH0gZnJvbSAnLi4vd29ya3NwYWNlcy9vZmZsaW5lJztcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0JztcblxuZXhwb3J0IGNvbnN0IHVzZUZpbHRlckZvbGRlcnNCeVdvcmtzcGFjZXMgPSAoXG4gIHF1ZXJ5OiBRdWVyeVByb3BzLFxuICB3YXRjaGVycz86IGFueVtdXG4pID0+IHtcbiAgY29uc3QgW2F2YWlsYWJsZUZvbGRlcnMsIHNldEF2YWlsYWJsZUZvbGRlcnNdID0gUmVhY3QudXNlU3RhdGU8c3RyaW5nW10+KFtdKTtcbiAgY29uc3QgW2ZpbHRlcmVkRm9sZGVycywgc2V0RmlsdGVyZWRGb2xkZXJzXSA9IFJlYWN0LnVzZVN0YXRlPFxuICAgIERpcmVjdG9yeUludGVyZmFjZVtdXG4gID4oW10pO1xuXG4gIGNvbnN0IGZpbHRlcmVkSW5uZXJGb2xkZXJzOiBzdHJpbmdbXSA9IFtdO1xuXG4gIGNvbnN0IHdvcmtzcGFjZSA9IHM7XG4gIGNvbnN0IGZvbGRlclBhdGhGcm9tUXVlcnkgPSBxdWVyeS5mb2xkZXJfcGF0aDtcbiAgY29uc3QgcGxvdF9zZWFyY2ggPSBxdWVyeS5wbG90X3NlYXJjaDtcblxuICBSZWFjdC51c2VFZmZlY3QoKCkgPT4ge1xuICAgIC8vZ2V0dGluZyBmb2xkZXJQYXRoIGJ5IHNlbGVjdGVkIHdvcmtzcGFjZVxuICAgIHdvcmtzcGFjZXMuZm9yRWFjaCgod29ya3NwYWNlRnJvbUxpc3Q6IGFueSkgPT4ge1xuICAgICAgd29ya3NwYWNlRnJvbUxpc3Qud29ya3NwYWNlcy5mb3JFYWNoKChvbmVXb3Jrc3BhY2U6IGFueSkgPT4ge1xuICAgICAgICBpZiAob25lV29ya3NwYWNlLmxhYmVsID09PSB3b3Jrc3BhY2UpIHtcbiAgICAgICAgICBzZXRBdmFpbGFibGVGb2xkZXJzKG9uZVdvcmtzcGFjZS5mb2xkZXJzUGF0aCk7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIH0pO1xuICB9LCBbd29ya3NwYWNlXSk7XG5cbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcbiAgICBpZiAod29ya3NwYWNlICYmICFmb2xkZXJQYXRoRnJvbVF1ZXJ5KSB7XG4gICAgICBjb25zdCBmaXJzdExheWVyRm9sZGVycyA9IF8udW5pcShcbiAgICAgICAgYXZhaWxhYmxlRm9sZGVycy5tYXAoKGZvbGRlcnNQYXRoOiBzdHJpbmcpID0+IHtcbiAgICAgICAgICBjb25zdCBmaXJzdExheWVyID0gZm9sZGVyc1BhdGguc3BsaXQoJy8nKVswXTtcbiAgICAgICAgICByZXR1cm4gZmlyc3RMYXllcjtcbiAgICAgICAgfSlcbiAgICAgICk7XG4gICAgICBjb25zdCBmb2xkZXJzX29iamVjdCA9IGZpcnN0TGF5ZXJGb2xkZXJzLm1hcCgoZm9sZGVyOiBzdHJpbmcpID0+IHtcbiAgICAgICAgcmV0dXJuIHsgc3ViZGlyOiBmb2xkZXIgfTtcbiAgICAgIH0pO1xuICAgICAgc2V0RmlsdGVyZWRGb2xkZXJzKGZvbGRlcnNfb2JqZWN0KTtcbiAgICB9IGVsc2UgaWYgKCEhd29ya3NwYWNlICYmICEhZm9sZGVyUGF0aEZyb21RdWVyeSkge1xuICAgICAgYXZhaWxhYmxlRm9sZGVycy5mb3JFYWNoKChmb2xkZXJzUGF0aDogc3RyaW5nKSA9PiB7XG4gICAgICAgIGNvbnN0IGZvbGRlclBhdGhGcm9tUXVlcnlXaXRob3V0Rmlyc3RTbGFzaCA9IHJlbW92ZUZpcnN0U2xhc2goXG4gICAgICAgICAgZm9sZGVyUGF0aEZyb21RdWVyeVxuICAgICAgICApOyAvL2lmIGZvbGRlclBhdGggaGFzIGEgc2xhc2ggaW4gdGhlIGJlZ2luaW5nLSByZW1vdmluZyBpdFxuICAgICAgICBjb25zdCBtYXRjaEJlZ2luaW5nSW5BdmFpbGFibGVGb2xkZXJzID0gZm9sZGVyc1BhdGguc2VhcmNoKFxuICAgICAgICAgIGZvbGRlclBhdGhGcm9tUXVlcnlXaXRob3V0Rmlyc3RTbGFzaFxuICAgICAgICApOyAvL3NlYXJjaGluZyBpbiBhdmFpbGFibGUgZm9sZGVycywgaXMgY2xpY2tlZCBmb2xkZXIgaXMgcGFydCBvZiBhdmFpbGFibGVGb2xkZXJzIHBhdGhcblxuICAgICAgICBpZiAobWF0Y2hCZWdpbmluZ0luQXZhaWxhYmxlRm9sZGVycyA+PSAwKSB7XG4gICAgICAgICAgLy8gaWYgc2VsZWN0ZWQgZm9sZGVyIGlzIGEgcGFydCBvZiBhdmFpbGFibGUgZm9sZGVyc3BhdGgsIHdlIHRyeWluZyB0byBnZXQgZnVydGhlciBsYXllciBmb2xkZXJzLlxuICAgICAgICAgIC8vbWF0Y2hFbmQgaXMgdGhlIGluZGV4LCB3aGljaCBpbmRpY2F0ZXMgdGhlIGVuZCBvZiBzZWxlY2VkIGZvbGRlciBwYXRoIHN0cmluZyAod2UgY2FuIHNlZSBpdCBpbiB1cmwpIGluIGF2YWlsYWJsZSBwYXRoXG4gICAgICAgICAgLy8gKGF2YWlsYmxlIHBhdGggd2Ugc2V0IHdpdGggc2V0QXZhaWxhYmxlRm9sZGVycyBhY3Rpb24pXG4gICAgICAgICAgY29uc3QgbWF0Y2hFbmQgPVxuICAgICAgICAgICAgbWF0Y2hCZWdpbmluZ0luQXZhaWxhYmxlRm9sZGVycyArXG4gICAgICAgICAgICBmb2xkZXJQYXRoRnJvbVF1ZXJ5V2l0aG91dEZpcnN0U2xhc2gubGVuZ3RoO1xuICAgICAgICAgIGNvbnN0IHJlc3RGb2xkZXJzID0gZm9sZGVyc1BhdGguc3Vic3RyaW5nKFxuICAgICAgICAgICAgbWF0Y2hFbmQsXG4gICAgICAgICAgICBmb2xkZXJzUGF0aC5sZW5ndGhcbiAgICAgICAgICApO1xuICAgICAgICAgIGNvbnN0IGZpcnN0TGF5ZXJGb2xkZXJPZlJlc3QgPSByZXN0Rm9sZGVycy5zcGxpdCgnLycpWzFdO1xuXG4gICAgICAgICAgLy9pZiBpdCBpcyB0aGUgbGFzdCBsYXllciBmaXJzdExheWVyRm9sZGVyT2ZSZXN0IHdpbGwgYmUgdW5kZWYuXG4gICAgICAgICAgLy8gaW4gdGhpcyBjYXNlIGZpbHRlcmVkSW5uZXJGb2xkZXJzIHdpbGwgYmUgZW1wdHkgYXJyYXlcbiAgICAgICAgICBpZiAoZmlyc3RMYXllckZvbGRlck9mUmVzdCkge1xuICAgICAgICAgICAgZmlsdGVyZWRJbm5lckZvbGRlcnMucHVzaChmaXJzdExheWVyRm9sZGVyT2ZSZXN0KTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgICAgY29uc3QgZm9sZGVyc19vYmplY3QgPSBmaWx0ZXJlZElubmVyRm9sZGVycy5tYXAoKGZvbGRlcjogc3RyaW5nKSA9PiB7XG4gICAgICAgIC8vbmVlZCB0byBoYXZlIHRoZSBzYW1lIGZvcm1hdCBhcyBkaXJlY3RvcmllcyBnb3QgZnJvbSBhcGkgb3IgYnkgcGxvdCBzZWFyY2hcbiAgICAgICAgcmV0dXJuIHsgc3ViZGlyOiBmb2xkZXIgfTtcbiAgICAgIH0pO1xuICAgICAgc2V0RmlsdGVyZWRGb2xkZXJzKGZvbGRlcnNfb2JqZWN0KTtcbiAgICB9XG4gIH0sIFtmb2xkZXJQYXRoRnJvbVF1ZXJ5LCBhdmFpbGFibGVGb2xkZXJzLCBwbG90X3NlYXJjaF0pO1xuXG4gIHJldHVybiB7IGZpbHRlcmVkRm9sZGVycyB9O1xufTtcbiJdLCJzb3VyY2VSb290IjoiIn0=