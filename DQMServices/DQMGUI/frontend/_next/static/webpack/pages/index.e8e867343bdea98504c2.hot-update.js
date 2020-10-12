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
  var workspace = query.workspaces;
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vaG9va3MvdXNlRmlsdGVyRm9sZGVyc0J5V29ya3NwYWNlLnRzeCJdLCJuYW1lcyI6WyJ1c2VGaWx0ZXJGb2xkZXJzQnlXb3Jrc3BhY2VzIiwicXVlcnkiLCJ3YXRjaGVycyIsIlJlYWN0IiwiYXZhaWxhYmxlRm9sZGVycyIsInNldEF2YWlsYWJsZUZvbGRlcnMiLCJmaWx0ZXJlZEZvbGRlcnMiLCJzZXRGaWx0ZXJlZEZvbGRlcnMiLCJmaWx0ZXJlZElubmVyRm9sZGVycyIsIndvcmtzcGFjZSIsIndvcmtzcGFjZXMiLCJmb2xkZXJQYXRoRnJvbVF1ZXJ5IiwiZm9sZGVyX3BhdGgiLCJwbG90X3NlYXJjaCIsImZvckVhY2giLCJ3b3Jrc3BhY2VGcm9tTGlzdCIsIm9uZVdvcmtzcGFjZSIsImxhYmVsIiwiZm9sZGVyc1BhdGgiLCJmaXJzdExheWVyRm9sZGVycyIsIl8iLCJ1bmlxIiwibWFwIiwiZmlyc3RMYXllciIsInNwbGl0IiwiZm9sZGVyc19vYmplY3QiLCJmb2xkZXIiLCJzdWJkaXIiLCJmb2xkZXJQYXRoRnJvbVF1ZXJ5V2l0aG91dEZpcnN0U2xhc2giLCJyZW1vdmVGaXJzdFNsYXNoIiwibWF0Y2hCZWdpbmluZ0luQXZhaWxhYmxlRm9sZGVycyIsInNlYXJjaCIsIm1hdGNoRW5kIiwibGVuZ3RoIiwicmVzdEZvbGRlcnMiLCJzdWJzdHJpbmciLCJmaXJzdExheWVyRm9sZGVyT2ZSZXN0IiwicHVzaCJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBTUE7QUFDQTtBQUdPLElBQU1BLDRCQUE0QixHQUFHLFNBQS9CQSw0QkFBK0IsQ0FDMUNDLEtBRDBDLEVBRTFDQyxRQUYwQyxFQUd2QztBQUFBOztBQUFBLHdCQUM2Q0MsOENBQUEsQ0FBeUIsRUFBekIsQ0FEN0M7QUFBQTtBQUFBLE1BQ0lDLGdCQURKO0FBQUEsTUFDc0JDLG1CQUR0Qjs7QUFBQSx5QkFFMkNGLDhDQUFBLENBRTVD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QU1OLDRCIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmU4ZTg2NzM0M2JkZWE5ODUwNGMyLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XG5pbXBvcnQgXyBmcm9tICdsb2Rhc2gnO1xuXG5pbXBvcnQge1xuICBRdWVyeVByb3BzLFxuICBEaXJlY3RvcnlJbnRlcmZhY2UsXG59IGZyb20gJy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcbmltcG9ydCB7IHJlbW92ZUZpcnN0U2xhc2ggfSBmcm9tICcuLi9jb21wb25lbnRzL3dvcmtzcGFjZXMvdXRpbHMnO1xuaW1wb3J0IHsgd29ya3NwYWNlcyB9IGZyb20gJy4uL3dvcmtzcGFjZXMvb2ZmbGluZSc7XG5pbXBvcnQgeyBzdG9yZSB9IGZyb20gJy4uL2NvbnRleHRzL2xlZnRTaWRlQ29udGV4dCc7XG5cbmV4cG9ydCBjb25zdCB1c2VGaWx0ZXJGb2xkZXJzQnlXb3Jrc3BhY2VzID0gKFxuICBxdWVyeTogUXVlcnlQcm9wcyxcbiAgd2F0Y2hlcnM/OiBhbnlbXVxuKSA9PiB7XG4gIGNvbnN0IFthdmFpbGFibGVGb2xkZXJzLCBzZXRBdmFpbGFibGVGb2xkZXJzXSA9IFJlYWN0LnVzZVN0YXRlPHN0cmluZ1tdPihbXSk7XG4gIGNvbnN0IFtmaWx0ZXJlZEZvbGRlcnMsIHNldEZpbHRlcmVkRm9sZGVyc10gPSBSZWFjdC51c2VTdGF0ZTxcbiAgICBEaXJlY3RvcnlJbnRlcmZhY2VbXVxuICA+KFtdKTtcblxuICBjb25zdCBmaWx0ZXJlZElubmVyRm9sZGVyczogc3RyaW5nW10gPSBbXTtcblxuICBjb25zdCB3b3Jrc3BhY2UgPSBxdWVyeS53b3Jrc3BhY2VzO1xuICBjb25zdCBmb2xkZXJQYXRoRnJvbVF1ZXJ5ID0gcXVlcnkuZm9sZGVyX3BhdGg7XG4gIGNvbnN0IHBsb3Rfc2VhcmNoID0gcXVlcnkucGxvdF9zZWFyY2g7XG5cbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcbiAgICAvL2dldHRpbmcgZm9sZGVyUGF0aCBieSBzZWxlY3RlZCB3b3Jrc3BhY2VcbiAgICB3b3Jrc3BhY2VzLmZvckVhY2goKHdvcmtzcGFjZUZyb21MaXN0OiBhbnkpID0+IHtcbiAgICAgIHdvcmtzcGFjZUZyb21MaXN0LndvcmtzcGFjZXMuZm9yRWFjaCgob25lV29ya3NwYWNlOiBhbnkpID0+IHtcbiAgICAgICAgaWYgKG9uZVdvcmtzcGFjZS5sYWJlbCA9PT0gd29ya3NwYWNlKSB7XG4gICAgICAgICAgc2V0QXZhaWxhYmxlRm9sZGVycyhvbmVXb3Jrc3BhY2UuZm9sZGVyc1BhdGgpO1xuICAgICAgICB9XG4gICAgICB9KTtcbiAgICB9KTtcbiAgfSwgW3dvcmtzcGFjZV0pO1xuXG4gIFJlYWN0LnVzZUVmZmVjdCgoKSA9PiB7XG4gICAgaWYgKHdvcmtzcGFjZSAmJiAhZm9sZGVyUGF0aEZyb21RdWVyeSkge1xuICAgICAgY29uc3QgZmlyc3RMYXllckZvbGRlcnMgPSBfLnVuaXEoXG4gICAgICAgIGF2YWlsYWJsZUZvbGRlcnMubWFwKChmb2xkZXJzUGF0aDogc3RyaW5nKSA9PiB7XG4gICAgICAgICAgY29uc3QgZmlyc3RMYXllciA9IGZvbGRlcnNQYXRoLnNwbGl0KCcvJylbMF07XG4gICAgICAgICAgcmV0dXJuIGZpcnN0TGF5ZXI7XG4gICAgICAgIH0pXG4gICAgICApO1xuICAgICAgY29uc3QgZm9sZGVyc19vYmplY3QgPSBmaXJzdExheWVyRm9sZGVycy5tYXAoKGZvbGRlcjogc3RyaW5nKSA9PiB7XG4gICAgICAgIHJldHVybiB7IHN1YmRpcjogZm9sZGVyIH07XG4gICAgICB9KTtcbiAgICAgIHNldEZpbHRlcmVkRm9sZGVycyhmb2xkZXJzX29iamVjdCk7XG4gICAgfSBlbHNlIGlmICghIXdvcmtzcGFjZSAmJiAhIWZvbGRlclBhdGhGcm9tUXVlcnkpIHtcbiAgICAgIGF2YWlsYWJsZUZvbGRlcnMuZm9yRWFjaCgoZm9sZGVyc1BhdGg6IHN0cmluZykgPT4ge1xuICAgICAgICBjb25zdCBmb2xkZXJQYXRoRnJvbVF1ZXJ5V2l0aG91dEZpcnN0U2xhc2ggPSByZW1vdmVGaXJzdFNsYXNoKFxuICAgICAgICAgIGZvbGRlclBhdGhGcm9tUXVlcnlcbiAgICAgICAgKTsgLy9pZiBmb2xkZXJQYXRoIGhhcyBhIHNsYXNoIGluIHRoZSBiZWdpbmluZy0gcmVtb3ZpbmcgaXRcbiAgICAgICAgY29uc3QgbWF0Y2hCZWdpbmluZ0luQXZhaWxhYmxlRm9sZGVycyA9IGZvbGRlcnNQYXRoLnNlYXJjaChcbiAgICAgICAgICBmb2xkZXJQYXRoRnJvbVF1ZXJ5V2l0aG91dEZpcnN0U2xhc2hcbiAgICAgICAgKTsgLy9zZWFyY2hpbmcgaW4gYXZhaWxhYmxlIGZvbGRlcnMsIGlzIGNsaWNrZWQgZm9sZGVyIGlzIHBhcnQgb2YgYXZhaWxhYmxlRm9sZGVycyBwYXRoXG5cbiAgICAgICAgaWYgKG1hdGNoQmVnaW5pbmdJbkF2YWlsYWJsZUZvbGRlcnMgPj0gMCkge1xuICAgICAgICAgIC8vIGlmIHNlbGVjdGVkIGZvbGRlciBpcyBhIHBhcnQgb2YgYXZhaWxhYmxlIGZvbGRlcnNwYXRoLCB3ZSB0cnlpbmcgdG8gZ2V0IGZ1cnRoZXIgbGF5ZXIgZm9sZGVycy5cbiAgICAgICAgICAvL21hdGNoRW5kIGlzIHRoZSBpbmRleCwgd2hpY2ggaW5kaWNhdGVzIHRoZSBlbmQgb2Ygc2VsZWNlZCBmb2xkZXIgcGF0aCBzdHJpbmcgKHdlIGNhbiBzZWUgaXQgaW4gdXJsKSBpbiBhdmFpbGFibGUgcGF0aFxuICAgICAgICAgIC8vIChhdmFpbGJsZSBwYXRoIHdlIHNldCB3aXRoIHNldEF2YWlsYWJsZUZvbGRlcnMgYWN0aW9uKVxuICAgICAgICAgIGNvbnN0IG1hdGNoRW5kID1cbiAgICAgICAgICAgIG1hdGNoQmVnaW5pbmdJbkF2YWlsYWJsZUZvbGRlcnMgK1xuICAgICAgICAgICAgZm9sZGVyUGF0aEZyb21RdWVyeVdpdGhvdXRGaXJzdFNsYXNoLmxlbmd0aDtcbiAgICAgICAgICBjb25zdCByZXN0Rm9sZGVycyA9IGZvbGRlcnNQYXRoLnN1YnN0cmluZyhcbiAgICAgICAgICAgIG1hdGNoRW5kLFxuICAgICAgICAgICAgZm9sZGVyc1BhdGgubGVuZ3RoXG4gICAgICAgICAgKTtcbiAgICAgICAgICBjb25zdCBmaXJzdExheWVyRm9sZGVyT2ZSZXN0ID0gcmVzdEZvbGRlcnMuc3BsaXQoJy8nKVsxXTtcblxuICAgICAgICAgIC8vaWYgaXQgaXMgdGhlIGxhc3QgbGF5ZXIgZmlyc3RMYXllckZvbGRlck9mUmVzdCB3aWxsIGJlIHVuZGVmLlxuICAgICAgICAgIC8vIGluIHRoaXMgY2FzZSBmaWx0ZXJlZElubmVyRm9sZGVycyB3aWxsIGJlIGVtcHR5IGFycmF5XG4gICAgICAgICAgaWYgKGZpcnN0TGF5ZXJGb2xkZXJPZlJlc3QpIHtcbiAgICAgICAgICAgIGZpbHRlcmVkSW5uZXJGb2xkZXJzLnB1c2goZmlyc3RMYXllckZvbGRlck9mUmVzdCk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9KTtcbiAgICAgIGNvbnN0IGZvbGRlcnNfb2JqZWN0ID0gZmlsdGVyZWRJbm5lckZvbGRlcnMubWFwKChmb2xkZXI6IHN0cmluZykgPT4ge1xuICAgICAgICAvL25lZWQgdG8gaGF2ZSB0aGUgc2FtZSBmb3JtYXQgYXMgZGlyZWN0b3JpZXMgZ290IGZyb20gYXBpIG9yIGJ5IHBsb3Qgc2VhcmNoXG4gICAgICAgIHJldHVybiB7IHN1YmRpcjogZm9sZGVyIH07XG4gICAgICB9KTtcbiAgICAgIHNldEZpbHRlcmVkRm9sZGVycyhmb2xkZXJzX29iamVjdCk7XG4gICAgfVxuICB9LCBbZm9sZGVyUGF0aEZyb21RdWVyeSwgYXZhaWxhYmxlRm9sZGVycywgcGxvdF9zZWFyY2hdKTtcblxuICByZXR1cm4geyBmaWx0ZXJlZEZvbGRlcnMgfTtcbn07XG4iXSwic291cmNlUm9vdCI6IiJ9