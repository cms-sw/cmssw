webpackHotUpdate_N_E("pages/index",{

/***/ "./containers/display/content/folders_and_plots_content.tsx":
/*!******************************************************************!*\
  !*** ./containers/display/content/folders_and_plots_content.tsx ***!
  \******************************************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _ant_design_icons__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! @ant-design/icons */ "./node_modules/@ant-design/icons/es/index.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! next/router */ "./node_modules/next/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_4__);
/* harmony import */ var lodash__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! lodash */ "./node_modules/lodash/lodash.js");
/* harmony import */ var lodash__WEBPACK_IMPORTED_MODULE_5___default = /*#__PURE__*/__webpack_require__.n(lodash__WEBPACK_IMPORTED_MODULE_5__);
/* harmony import */ var _components_plots_zoomedPlots__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../../components/plots/zoomedPlots */ "./components/plots/zoomedPlots/index.tsx");
/* harmony import */ var _components_viewDetailsMenu__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../../components/viewDetailsMenu */ "./components/viewDetailsMenu/index.tsx");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../styledComponents */ "./containers/display/styledComponents.tsx");
/* harmony import */ var _folderPath__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ./folderPath */ "./containers/display/content/folderPath.tsx");
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../utils */ "./containers/display/utils.ts");
/* harmony import */ var _components_styledComponents__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ../../../components/styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _hooks_useFilterFolders__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../../hooks/useFilterFolders */ "./hooks/useFilterFolders.tsx");
/* harmony import */ var _components_settings__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ../../../components/settings */ "./components/settings/index.tsx");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_14__ = __webpack_require__(/*! ../../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _display_folders_or_plots__WEBPACK_IMPORTED_MODULE_15__ = __webpack_require__(/*! ./display_folders_or_plots */ "./containers/display/content/display_folders_or_plots.tsx");
/* harmony import */ var _components_usefulLinks__WEBPACK_IMPORTED_MODULE_16__ = __webpack_require__(/*! ../../../components/usefulLinks */ "./components/usefulLinks/index.tsx");
/* harmony import */ var _components_workspaces__WEBPACK_IMPORTED_MODULE_17__ = __webpack_require__(/*! ../../../components/workspaces */ "./components/workspaces/index.tsx");
/* harmony import */ var _components_plots_plot_plotSearch__WEBPACK_IMPORTED_MODULE_18__ = __webpack_require__(/*! ../../../components/plots/plot/plotSearch */ "./components/plots/plot/plotSearch/index.tsx");


var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/containers/display/content/folders_and_plots_content.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_1___default.a.createElement;



















var Content = function Content(_ref) {
  _s();

  var folder_path = _ref.folder_path,
      run_number = _ref.run_number,
      dataset_name = _ref.dataset_name;

  var _useContext = Object(react__WEBPACK_IMPORTED_MODULE_1__["useContext"])(_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_14__["store"]),
      viewPlotsPosition = _useContext.viewPlotsPosition,
      proportion = _useContext.proportion,
      updated_by_not_older_than = _useContext.updated_by_not_older_than;

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_4__["useRouter"])();
  var query = router.query;
  var params = {
    run_number: run_number,
    dataset_name: dataset_name,
    folders_path: folder_path,
    notOlderThan: updated_by_not_older_than,
    plot_search: query.plot_search
  };

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_1__["useState"])(false),
      openSettings = _useState[0],
      toggleSettingsModal = _useState[1];

  var selectedPlots = query.selected_plots; //filtering directories by selected workspace

  var _useFilterFolders = Object(_hooks_useFilterFolders__WEBPACK_IMPORTED_MODULE_12__["useFilterFolders"])(query, params, updated_by_not_older_than),
      foldersByPlotSearch = _useFilterFolders.foldersByPlotSearch,
      plots = _useFilterFolders.plots,
      isLoading = _useFilterFolders.isLoading,
      errors = _useFilterFolders.errors;

  var plots_with_layouts = plots.filter(function (plot) {
    return plot.hasOwnProperty('layout');
  });
  var plots_grouped_by_layouts = Object(lodash__WEBPACK_IMPORTED_MODULE_5__["chain"])(plots_with_layouts).sortBy('layout').groupBy('layout').value();
  var filteredFolders = foldersByPlotSearch ? foldersByPlotSearch : [];
  var selected_plots = Object(_utils__WEBPACK_IMPORTED_MODULE_10__["getSelectedPlots"])(selectedPlots, plots);

  var changeFolderPathByBreadcrumb = function changeFolderPathByBreadcrumb(parameters) {
    return Object(_utils__WEBPACK_IMPORTED_MODULE_10__["changeRouter"])(Object(_utils__WEBPACK_IMPORTED_MODULE_10__["getChangedQueryParams"])(parameters, query));
  };

  var plotsAreaRef = react__WEBPACK_IMPORTED_MODULE_1___default.a.useRef(null);

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_1___default.a.useState(0),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState, 2),
      plotsAreaWidth = _React$useState2[0],
      setPlotsAreaWidth = _React$useState2[1];

  react__WEBPACK_IMPORTED_MODULE_1___default.a.useEffect(function () {
    if (plotsAreaRef.current) {
      setPlotsAreaWidth(plotsAreaRef.current.clientWidth);
    }
  }, [plotsAreaRef.current]);
  return __jsx(react__WEBPACK_IMPORTED_MODULE_1___default.a.Fragment, null, __jsx(_components_styledComponents__WEBPACK_IMPORTED_MODULE_11__["CustomRow"], {
    space: '2',
    width: "100%",
    justifycontent: "space-between",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 102,
      columnNumber: 7
    }
  }, __jsx(_components_settings__WEBPACK_IMPORTED_MODULE_13__["SettingsModal"], {
    openSettings: openSettings,
    toggleSettingsModal: toggleSettingsModal,
    isAnyPlotSelected: selected_plots.length === 0,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 103,
      columnNumber: 9
    }
  }), __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Col"], {
    style: {
      padding: 8
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 108,
      columnNumber: 9
    }
  }, __jsx(_folderPath__WEBPACK_IMPORTED_MODULE_9__["FolderPath"], {
    folder_path: folder_path,
    changeFolderPathByBreadcrumb: changeFolderPathByBreadcrumb,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 109,
      columnNumber: 11
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Row"], {
    gutter: 16,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 111,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 112,
      columnNumber: 11
    }
  }, __jsx(_components_workspaces__WEBPACK_IMPORTED_MODULE_17__["default"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 113,
      columnNumber: 13
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 115,
      columnNumber: 11
    }
  }, __jsx(_components_plots_plot_plotSearch__WEBPACK_IMPORTED_MODULE_18__["PlotSearch"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 116,
      columnNumber: 13
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 118,
      columnNumber: 11
    }
  }, __jsx(_components_usefulLinks__WEBPACK_IMPORTED_MODULE_16__["UsefulLinks"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 119,
      columnNumber: 13
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 121,
      columnNumber: 11
    }
  }, __jsx(_components_styledComponents__WEBPACK_IMPORTED_MODULE_11__["StyledSecondaryButton"], {
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_3__["SettingOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 123,
        columnNumber: 21
      }
    }),
    onClick: function onClick() {
      return toggleSettingsModal(true);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 122,
      columnNumber: 13
    }
  }, "Settings")))), __jsx(_components_styledComponents__WEBPACK_IMPORTED_MODULE_11__["CustomRow"], {
    width: "100%",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 131,
      columnNumber: 7
    }
  }, __jsx(_components_viewDetailsMenu__WEBPACK_IMPORTED_MODULE_7__["ViewDetailsMenu"], {
    plotsAreaWidth: plotsAreaWidth,
    selected_plots: selected_plots.length > 0,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 132,
      columnNumber: 9
    }
  })), __jsx(react__WEBPACK_IMPORTED_MODULE_1___default.a.Fragment, null, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_8__["DivWrapper"], {
    selectedPlots: selected_plots.length > 0,
    position: viewPlotsPosition,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 135,
      columnNumber: 9
    }
  }, __jsx(_display_folders_or_plots__WEBPACK_IMPORTED_MODULE_15__["DisplayFordersOrPlots"], {
    plotsAreaRef: plotsAreaRef,
    plots: plots,
    selected_plots: selected_plots,
    plots_grouped_by_layouts: plots_grouped_by_layouts,
    isLoading: isLoading,
    viewPlotsPosition: viewPlotsPosition,
    proportion: proportion,
    errors: errors,
    filteredFolders: filteredFolders,
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 139,
      columnNumber: 11
    }
  }), selected_plots.length > 0 && errors.length === 0 && __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_8__["ZoomedPlotsWrapper"], {
    any_selected_plots: selected_plots.length && errors.length === 0,
    proportion: proportion,
    position: viewPlotsPosition,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 152,
      columnNumber: 13
    }
  }, __jsx(_components_plots_zoomedPlots__WEBPACK_IMPORTED_MODULE_6__["ZoomedPlots"], {
    selected_plots: selected_plots,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 157,
      columnNumber: 15
    }
  })))));
};

_s(Content, "WM2LuWRNDQWAmH4Yi+BN0nVunks=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_4__["useRouter"], _hooks_useFilterFolders__WEBPACK_IMPORTED_MODULE_12__["useFilterFolders"]];
});

_c = Content;
/* harmony default export */ __webpack_exports__["default"] = (Content);

var _c;

$RefreshReg$(_c, "Content");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGFpbmVycy9kaXNwbGF5L2NvbnRlbnQvZm9sZGVyc19hbmRfcGxvdHNfY29udGVudC50c3giXSwibmFtZXMiOlsiQ29udGVudCIsImZvbGRlcl9wYXRoIiwicnVuX251bWJlciIsImRhdGFzZXRfbmFtZSIsInVzZUNvbnRleHQiLCJzdG9yZSIsInZpZXdQbG90c1Bvc2l0aW9uIiwicHJvcG9ydGlvbiIsInVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4iLCJyb3V0ZXIiLCJ1c2VSb3V0ZXIiLCJxdWVyeSIsInBhcmFtcyIsImZvbGRlcnNfcGF0aCIsIm5vdE9sZGVyVGhhbiIsInBsb3Rfc2VhcmNoIiwidXNlU3RhdGUiLCJvcGVuU2V0dGluZ3MiLCJ0b2dnbGVTZXR0aW5nc01vZGFsIiwic2VsZWN0ZWRQbG90cyIsInNlbGVjdGVkX3Bsb3RzIiwidXNlRmlsdGVyRm9sZGVycyIsImZvbGRlcnNCeVBsb3RTZWFyY2giLCJwbG90cyIsImlzTG9hZGluZyIsImVycm9ycyIsInBsb3RzX3dpdGhfbGF5b3V0cyIsImZpbHRlciIsInBsb3QiLCJoYXNPd25Qcm9wZXJ0eSIsInBsb3RzX2dyb3VwZWRfYnlfbGF5b3V0cyIsImNoYWluIiwic29ydEJ5IiwiZ3JvdXBCeSIsInZhbHVlIiwiZmlsdGVyZWRGb2xkZXJzIiwiZ2V0U2VsZWN0ZWRQbG90cyIsImNoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWIiLCJwYXJhbWV0ZXJzIiwiY2hhbmdlUm91dGVyIiwiZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zIiwicGxvdHNBcmVhUmVmIiwiUmVhY3QiLCJ1c2VSZWYiLCJwbG90c0FyZWFXaWR0aCIsInNldFBsb3RzQXJlYVdpZHRoIiwidXNlRWZmZWN0IiwiY3VycmVudCIsImNsaWVudFdpZHRoIiwibGVuZ3RoIiwicGFkZGluZyJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7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QU0sQ0FBQ3dCLE1BQVAsS0FBa0IsQ0FEakU7QUFFRSxjQUFVLEVBQUUxQyxVQUZkO0FBR0UsWUFBUSxFQUFFRCxpQkFIWjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBS0UsTUFBQyx5RUFBRDtBQUFhLGtCQUFjLEVBQUVjLGNBQTdCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFMRixDQWpCSixDQURGLENBakNGLENBREY7QUFnRUQsQ0FuSEQ7O0dBQU1wQixPO1VBV1dVLHFELEVBZTJDVyx5RTs7O0tBMUJ0RHJCLE87QUFxSFNBLHNFQUFmIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjhiOWExYWU3OTQyZTA0NjExMzJmLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgRkMsIHVzZVN0YXRlLCB1c2VDb250ZXh0IH0gZnJvbSAncmVhY3QnO1xyXG5pbXBvcnQgeyBDb2wsIFJvdyB9IGZyb20gJ2FudGQnO1xyXG5pbXBvcnQgeyBTZXR0aW5nT3V0bGluZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XHJcbmltcG9ydCB7IHVzZVJvdXRlciB9IGZyb20gJ25leHQvcm91dGVyJztcclxuaW1wb3J0IHsgY2hhaW4gfSBmcm9tICdsb2Rhc2gnO1xyXG5cclxuaW1wb3J0IHsgUGxvdERhdGFQcm9wcywgUXVlcnlQcm9wcyB9IGZyb20gJy4uL2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyBab29tZWRQbG90cyB9IGZyb20gJy4uLy4uLy4uL2NvbXBvbmVudHMvcGxvdHMvem9vbWVkUGxvdHMnO1xyXG5pbXBvcnQgeyBWaWV3RGV0YWlsc01lbnUgfSBmcm9tICcuLi8uLi8uLi9jb21wb25lbnRzL3ZpZXdEZXRhaWxzTWVudSc7XHJcbmltcG9ydCB7IERpdldyYXBwZXIsIFpvb21lZFBsb3RzV3JhcHBlciB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyBGb2xkZXJQYXRoIH0gZnJvbSAnLi9mb2xkZXJQYXRoJztcclxuaW1wb3J0IHsgY2hhbmdlUm91dGVyLCBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMsIGdldFNlbGVjdGVkUGxvdHMgfSBmcm9tICcuLi91dGlscyc7XHJcbmltcG9ydCB7XHJcbiAgQ3VzdG9tUm93LFxyXG4gIFN0eWxlZFNlY29uZGFyeUJ1dHRvbixcclxufSBmcm9tICcuLi8uLi8uLi9jb21wb25lbnRzL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyB1c2VGaWx0ZXJGb2xkZXJzIH0gZnJvbSAnLi4vLi4vLi4vaG9va3MvdXNlRmlsdGVyRm9sZGVycyc7XHJcbmltcG9ydCB7IFNldHRpbmdzTW9kYWwgfSBmcm9tICcuLi8uLi8uLi9jb21wb25lbnRzL3NldHRpbmdzJztcclxuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnO1xyXG5pbXBvcnQgeyBEaXNwbGF5Rm9yZGVyc09yUGxvdHMgfSBmcm9tICcuL2Rpc3BsYXlfZm9sZGVyc19vcl9wbG90cyc7XHJcbmltcG9ydCB7IFVzZWZ1bExpbmtzIH0gZnJvbSAnLi4vLi4vLi4vY29tcG9uZW50cy91c2VmdWxMaW5rcyc7XHJcbmltcG9ydCB7IFBhcnNlZFVybFF1ZXJ5SW5wdXQgfSBmcm9tICdxdWVyeXN0cmluZyc7XHJcbmltcG9ydCBXb3Jrc3BhY2VzIGZyb20gJy4uLy4uLy4uL2NvbXBvbmVudHMvd29ya3NwYWNlcyc7XHJcbmltcG9ydCB7IFBsb3RTZWFyY2ggfSBmcm9tICcuLi8uLi8uLi9jb21wb25lbnRzL3Bsb3RzL3Bsb3QvcGxvdFNlYXJjaCc7XHJcblxyXG5leHBvcnQgaW50ZXJmYWNlIFBsb3RJbnRlcmZhY2Uge1xyXG4gIG9iaj86IHN0cmluZztcclxuICBuYW1lPzogc3RyaW5nO1xyXG4gIHBhdGg6IHN0cmluZztcclxuICBjb250ZW50OiBhbnk7XHJcbiAgcHJvcGVydGllczogYW55O1xyXG4gIGxheW91dD86IHN0cmluZztcclxuICByZXBvcnQ/OiBhbnk7XHJcbiAgcXJlc3VsdHM/OiBbXTtcclxuICBxdHN0YXR1c2VzPzogW107XHJcbn1cclxuXHJcbmV4cG9ydCBpbnRlcmZhY2UgRm9sZGVyUGF0aEJ5QnJlYWRjcnVtYlByb3BzIHtcclxuICBmb2xkZXJfcGF0aDogc3RyaW5nO1xyXG4gIG5hbWU6IHN0cmluZztcclxufVxyXG5cclxuaW50ZXJmYWNlIEZvbGRlclByb3BzIHtcclxuICBmb2xkZXJfcGF0aD86IHN0cmluZztcclxuICBydW5fbnVtYmVyOiBzdHJpbmc7XHJcbiAgZGF0YXNldF9uYW1lOiBzdHJpbmc7XHJcbn1cclxuXHJcbmNvbnN0IENvbnRlbnQ6IEZDPEZvbGRlclByb3BzPiA9ICh7XHJcbiAgZm9sZGVyX3BhdGgsXHJcbiAgcnVuX251bWJlcixcclxuICBkYXRhc2V0X25hbWUsXHJcbn0pID0+IHtcclxuICBjb25zdCB7XHJcbiAgICB2aWV3UGxvdHNQb3NpdGlvbixcclxuICAgIHByb3BvcnRpb24sXHJcbiAgICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuLFxyXG4gIH0gPSB1c2VDb250ZXh0KHN0b3JlKTtcclxuXHJcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XHJcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XHJcblxyXG4gIGNvbnN0IHBhcmFtcyA9IHtcclxuICAgIHJ1bl9udW1iZXI6IHJ1bl9udW1iZXIsXHJcbiAgICBkYXRhc2V0X25hbWU6IGRhdGFzZXRfbmFtZSxcclxuICAgIGZvbGRlcnNfcGF0aDogZm9sZGVyX3BhdGgsXHJcbiAgICBub3RPbGRlclRoYW46IHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4sXHJcbiAgICBwbG90X3NlYXJjaDogcXVlcnkucGxvdF9zZWFyY2gsXHJcbiAgfTtcclxuXHJcbiAgY29uc3QgW29wZW5TZXR0aW5ncywgdG9nZ2xlU2V0dGluZ3NNb2RhbF0gPSB1c2VTdGF0ZShmYWxzZSk7XHJcblxyXG4gIGNvbnN0IHNlbGVjdGVkUGxvdHMgPSBxdWVyeS5zZWxlY3RlZF9wbG90cztcclxuICAvL2ZpbHRlcmluZyBkaXJlY3RvcmllcyBieSBzZWxlY3RlZCB3b3Jrc3BhY2VcclxuICBjb25zdCB7IGZvbGRlcnNCeVBsb3RTZWFyY2gsIHBsb3RzLCBpc0xvYWRpbmcsIGVycm9ycyB9ID0gdXNlRmlsdGVyRm9sZGVycyhcclxuICAgIHF1ZXJ5LFxyXG4gICAgcGFyYW1zLFxyXG4gICAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhblxyXG4gICk7XHJcbiAgY29uc3QgcGxvdHNfd2l0aF9sYXlvdXRzID0gcGxvdHMuZmlsdGVyKChwbG90KSA9PiBwbG90Lmhhc093blByb3BlcnR5KCdsYXlvdXQnKSlcclxuICB2YXIgcGxvdHNfZ3JvdXBlZF9ieV9sYXlvdXRzID0gY2hhaW4ocGxvdHNfd2l0aF9sYXlvdXRzKS5zb3J0QnkoJ2xheW91dCcpLmdyb3VwQnkoJ2xheW91dCcpLnZhbHVlKClcclxuICBjb25zdCBmaWx0ZXJlZEZvbGRlcnM6IGFueVtdID0gZm9sZGVyc0J5UGxvdFNlYXJjaCA/IGZvbGRlcnNCeVBsb3RTZWFyY2ggOiBbXTtcclxuICBjb25zdCBzZWxlY3RlZF9wbG90czogUGxvdERhdGFQcm9wc1tdID0gZ2V0U2VsZWN0ZWRQbG90cyhcclxuICAgIHNlbGVjdGVkUGxvdHMsXHJcbiAgICBwbG90c1xyXG4gICk7XHJcblxyXG4gIGNvbnN0IGNoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWIgPSAocGFyYW1ldGVyczogUGFyc2VkVXJsUXVlcnlJbnB1dCkgPT5cclxuICAgIGNoYW5nZVJvdXRlcihnZXRDaGFuZ2VkUXVlcnlQYXJhbXMocGFyYW1ldGVycywgcXVlcnkpKTtcclxuXHJcbiAgY29uc3QgcGxvdHNBcmVhUmVmID0gUmVhY3QudXNlUmVmPGFueT4obnVsbClcclxuICBjb25zdCBbcGxvdHNBcmVhV2lkdGgsIHNldFBsb3RzQXJlYVdpZHRoXSA9IFJlYWN0LnVzZVN0YXRlKDApXHJcblxyXG4gIFJlYWN0LnVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBpZiAocGxvdHNBcmVhUmVmLmN1cnJlbnQpIHtcclxuICAgICAgc2V0UGxvdHNBcmVhV2lkdGgocGxvdHNBcmVhUmVmLmN1cnJlbnQuY2xpZW50V2lkdGgpXHJcbiAgICB9XHJcbiAgfSwgW3Bsb3RzQXJlYVJlZi5jdXJyZW50XSlcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDw+XHJcbiAgICAgIDxDdXN0b21Sb3cgc3BhY2U9eycyJ30gd2lkdGg9XCIxMDAlXCIganVzdGlmeWNvbnRlbnQ9XCJzcGFjZS1iZXR3ZWVuXCI+XHJcbiAgICAgICAgPFNldHRpbmdzTW9kYWxcclxuICAgICAgICAgIG9wZW5TZXR0aW5ncz17b3BlblNldHRpbmdzfVxyXG4gICAgICAgICAgdG9nZ2xlU2V0dGluZ3NNb2RhbD17dG9nZ2xlU2V0dGluZ3NNb2RhbH1cclxuICAgICAgICAgIGlzQW55UGxvdFNlbGVjdGVkPXtzZWxlY3RlZF9wbG90cy5sZW5ndGggPT09IDB9XHJcbiAgICAgICAgLz5cclxuICAgICAgICA8Q29sIHN0eWxlPXt7IHBhZGRpbmc6IDggfX0+XHJcbiAgICAgICAgICA8Rm9sZGVyUGF0aCBmb2xkZXJfcGF0aD17Zm9sZGVyX3BhdGh9IGNoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWI9e2NoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWJ9IC8+XHJcbiAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgPFJvdyBndXR0ZXI9ezE2fT5cclxuICAgICAgICAgIDxDb2w+XHJcbiAgICAgICAgICAgIDxXb3Jrc3BhY2VzIC8+XHJcbiAgICAgICAgICA8L0NvbD5cclxuICAgICAgICAgIDxDb2w+XHJcbiAgICAgICAgICAgIDxQbG90U2VhcmNoIC8+XHJcbiAgICAgICAgICA8L0NvbD5cclxuICAgICAgICAgIDxDb2w+XHJcbiAgICAgICAgICAgIDxVc2VmdWxMaW5rcyAvPlxyXG4gICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICA8Q29sPlxyXG4gICAgICAgICAgICA8U3R5bGVkU2Vjb25kYXJ5QnV0dG9uXHJcbiAgICAgICAgICAgICAgaWNvbj17PFNldHRpbmdPdXRsaW5lZCAvPn1cclxuICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB0b2dnbGVTZXR0aW5nc01vZGFsKHRydWUpfVxyXG4gICAgICAgICAgICA+XHJcbiAgICAgICAgICAgICAgU2V0dGluZ3NcclxuICAgICAgICAgIDwvU3R5bGVkU2Vjb25kYXJ5QnV0dG9uPlxyXG4gICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgPC9Sb3c+XHJcbiAgICAgIDwvQ3VzdG9tUm93PlxyXG4gICAgICA8Q3VzdG9tUm93IHdpZHRoPVwiMTAwJVwiPlxyXG4gICAgICAgIDxWaWV3RGV0YWlsc01lbnUgcGxvdHNBcmVhV2lkdGg9e3Bsb3RzQXJlYVdpZHRofSBzZWxlY3RlZF9wbG90cz17c2VsZWN0ZWRfcGxvdHMubGVuZ3RoID4gMH0gLz5cclxuICAgICAgPC9DdXN0b21Sb3c+XHJcbiAgICAgIDw+XHJcbiAgICAgICAgPERpdldyYXBwZXJcclxuICAgICAgICAgIHNlbGVjdGVkUGxvdHM9e3NlbGVjdGVkX3Bsb3RzLmxlbmd0aCA+IDB9XHJcbiAgICAgICAgICBwb3NpdGlvbj17dmlld1Bsb3RzUG9zaXRpb259XHJcbiAgICAgICAgPlxyXG4gICAgICAgICAgPERpc3BsYXlGb3JkZXJzT3JQbG90c1xyXG4gICAgICAgICAgICBwbG90c0FyZWFSZWY9e3Bsb3RzQXJlYVJlZn1cclxuICAgICAgICAgICAgcGxvdHM9e3Bsb3RzfVxyXG4gICAgICAgICAgICBzZWxlY3RlZF9wbG90cz17c2VsZWN0ZWRfcGxvdHN9XHJcbiAgICAgICAgICAgIHBsb3RzX2dyb3VwZWRfYnlfbGF5b3V0cz17cGxvdHNfZ3JvdXBlZF9ieV9sYXlvdXRzfVxyXG4gICAgICAgICAgICBpc0xvYWRpbmc9e2lzTG9hZGluZ31cclxuICAgICAgICAgICAgdmlld1Bsb3RzUG9zaXRpb249e3ZpZXdQbG90c1Bvc2l0aW9ufVxyXG4gICAgICAgICAgICBwcm9wb3J0aW9uPXtwcm9wb3J0aW9ufVxyXG4gICAgICAgICAgICBlcnJvcnM9e2Vycm9yc31cclxuICAgICAgICAgICAgZmlsdGVyZWRGb2xkZXJzPXtmaWx0ZXJlZEZvbGRlcnN9XHJcbiAgICAgICAgICAgIHF1ZXJ5PXtxdWVyeX1cclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgICB7c2VsZWN0ZWRfcGxvdHMubGVuZ3RoID4gMCAmJiBlcnJvcnMubGVuZ3RoID09PSAwICYmIChcclxuICAgICAgICAgICAgPFpvb21lZFBsb3RzV3JhcHBlclxyXG4gICAgICAgICAgICAgIGFueV9zZWxlY3RlZF9wbG90cz17c2VsZWN0ZWRfcGxvdHMubGVuZ3RoICYmIGVycm9ycy5sZW5ndGggPT09IDB9XHJcbiAgICAgICAgICAgICAgcHJvcG9ydGlvbj17cHJvcG9ydGlvbn1cclxuICAgICAgICAgICAgICBwb3NpdGlvbj17dmlld1Bsb3RzUG9zaXRpb259XHJcbiAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICA8Wm9vbWVkUGxvdHMgc2VsZWN0ZWRfcGxvdHM9e3NlbGVjdGVkX3Bsb3RzfSAvPlxyXG4gICAgICAgICAgICA8L1pvb21lZFBsb3RzV3JhcHBlcj5cclxuICAgICAgICAgICl9XHJcbiAgICAgICAgPC9EaXZXcmFwcGVyPlxyXG4gICAgICA8Lz5cclxuICAgIDwvPlxyXG4gICk7XHJcbn07XHJcblxyXG5leHBvcnQgZGVmYXVsdCBDb250ZW50O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9