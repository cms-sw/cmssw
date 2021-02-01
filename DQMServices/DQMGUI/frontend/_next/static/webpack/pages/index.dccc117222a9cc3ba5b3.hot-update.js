webpackHotUpdate_N_E("pages/index",{

/***/ "./components/plots/plot/plotSearch/index.tsx":
/*!****************************************************!*\
  !*** ./components/plots/plot/plotSearch/index.tsx ***!
  \****************************************************/
/*! exports provided: PlotSearch */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "PlotSearch", function() { return PlotSearch; });
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! antd/lib/form/Form */ "./node_modules/antd/lib/form/Form.js");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! next/router */ "./node_modules/next/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../../../containers/display/utils */ "./containers/display/utils.ts");


var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/plots/plot/plotSearch/index.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_1__["createElement"];





var PlotSearch = function PlotSearch() {
  _s();

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"])();
  var query = router.query;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_1__["useState"](query.plot_search),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState, 2),
      plotName = _React$useState2[0],
      setPlotName = _React$useState2[1];

  react__WEBPACK_IMPORTED_MODULE_1__["useEffect"](function () {
    if (query.plot_search !== plotName) {
      var params = Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_5__["getChangedQueryParams"])({
        plot_search: plotName
      }, query);
      Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_5__["changeRouter"])(params);
    }
  }, [plotName]);
  return react__WEBPACK_IMPORTED_MODULE_1__["useMemo"](function () {
    return __jsx(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_2___default.a, {
      onChange: function onChange(e) {
        return setPlotName(e.target.value);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 29,
        columnNumber: 7
      }
    }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 30,
        columnNumber: 9
      }
    }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledSearch"], {
      defaultValue: query.plot_search,
      id: "plot_search",
      placeholder: "Enter plot name",
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 31,
        columnNumber: 11
      }
    })));
  }, [plotName]);
};

_s(PlotSearch, "qUuwOtWUsWURNKw3w2PjYEO5WgU=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"]];
});

_c = PlotSearch;

var _c;

$RefreshReg$(_c, "PlotSearch");

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ }),

/***/ "./components/workspaces/index.tsx":
/*!*****************************************!*\
  !*** ./components/workspaces/index.tsx ***!
  \*****************************************/
/*! exports provided: default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/regenerator */ "./node_modules/@babel/runtime/regenerator/index.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/asyncToGenerator */ "./node_modules/@babel/runtime/helpers/esm/asyncToGenerator.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _workspaces_offline__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../workspaces/offline */ "./workspaces/offline.ts");
/* harmony import */ var _workspaces_online__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../workspaces/online */ "./workspaces/online.ts");
/* harmony import */ var _viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../viewDetailsMenu/styledComponents */ "./components/viewDetailsMenu/styledComponents.tsx");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! antd/lib/form/Form */ "./node_modules/antd/lib/form/Form.js");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8___default = /*#__PURE__*/__webpack_require__.n(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8__);
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! next/router */ "./node_modules/next/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_10___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_10__);
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ./utils */ "./components/workspaces/utils.ts");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_13__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");




var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/workspaces/index.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_3__["createElement"];











var TabPane = antd__WEBPACK_IMPORTED_MODULE_4__["Tabs"].TabPane;

var Workspaces = function Workspaces() {
  _s();

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_3__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_13__["store"]),
      workspace = _React$useContext.workspace,
      setWorkspace = _React$useContext.setWorkspace;

  var workspaces = _config_config__WEBPACK_IMPORTED_MODULE_12__["functions_config"].mode === 'ONLINE' ? _workspaces_online__WEBPACK_IMPORTED_MODULE_6__["workspaces"] : _workspaces_offline__WEBPACK_IMPORTED_MODULE_5__["workspaces"];
  var initialWorkspace = _config_config__WEBPACK_IMPORTED_MODULE_12__["functions_config"].mode === 'ONLINE' ? workspaces[0].workspaces[1].label : workspaces[0].workspaces[3].label;
  react__WEBPACK_IMPORTED_MODULE_3__["useEffect"](function () {
    setWorkspace(initialWorkspace);
    return function () {
      return setWorkspace(initialWorkspace);
    };
  }, []);
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_10__["useRouter"])();
  var query = router.query;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_3__["useState"](false),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__["default"])(_React$useState, 2),
      openWorkspaces = _React$useState2[0],
      toggleWorkspaces = _React$useState2[1]; // make a workspace set from context


  return __jsx(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_8___default.a, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 42,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_9__["StyledFormItem"], {
    labelcolor: "white",
    label: "Workspace",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 43,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Button"], {
    onClick: function onClick() {
      toggleWorkspaces(!openWorkspaces);
    },
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 44,
      columnNumber: 9
    }
  }, workspace), __jsx(_viewDetailsMenu_styledComponents__WEBPACK_IMPORTED_MODULE_7__["StyledModal"], {
    title: "Workspaces",
    visible: openWorkspaces,
    onCancel: function onCancel() {
      return toggleWorkspaces(false);
    },
    footer: null,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 52,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Tabs"], {
    defaultActiveKey: "1",
    type: "card",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 58,
      columnNumber: 11
    }
  }, workspaces.map(function (workspace) {
    return __jsx(TabPane, {
      key: workspace.label,
      tab: workspace.label,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 60,
        columnNumber: 15
      }
    }, workspace.workspaces.map(function (subWorkspace) {
      return __jsx(antd__WEBPACK_IMPORTED_MODULE_4__["Button"], {
        key: subWorkspace.label,
        type: "link",
        onClick: /*#__PURE__*/Object(_babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__["default"])( /*#__PURE__*/_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.mark(function _callee() {
          return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.wrap(function _callee$(_context) {
            while (1) {
              switch (_context.prev = _context.next) {
                case 0:
                  setWorkspace(subWorkspace.label);
                  toggleWorkspaces(!openWorkspaces); //if workspace is selected, folder_path in query is set to ''. Then we can regonize
                  //that workspace is selected, and wee need to filter the forst layer of folders.

                  _context.next = 4;
                  return Object(_utils__WEBPACK_IMPORTED_MODULE_11__["setWorkspaceToQuery"])(query, subWorkspace.label);

                case 4:
                case "end":
                  return _context.stop();
              }
            }
          }, _callee);
        })),
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 62,
          columnNumber: 19
        }
      }, subWorkspace.label);
    }));
  })))));
};

_s(Workspaces, "9wsb3E7mFlyFmQpi1Uvfk2BcVak=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_10__["useRouter"]];
});

_c = Workspaces;
/* harmony default export */ __webpack_exports__["default"] = (Workspaces);

var _c;

$RefreshReg$(_c, "Workspaces");

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

/***/ }),

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
  }, __jsx(_components_usefulLinks__WEBPACK_IMPORTED_MODULE_16__["UsefulLinks"], {
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
  }, __jsx(_components_styledComponents__WEBPACK_IMPORTED_MODULE_11__["StyledSecondaryButton"], {
    icon: __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_3__["SettingOutlined"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 117,
        columnNumber: 21
      }
    }),
    onClick: function onClick() {
      return toggleSettingsModal(true);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 116,
      columnNumber: 13
    }
  }, "Settings")), __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 123,
      columnNumber: 11
    }
  }, __jsx(_components_workspaces__WEBPACK_IMPORTED_MODULE_17__["default"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 124,
      columnNumber: 13
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_2__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 126,
      columnNumber: 11
    }
  }, __jsx(_components_plots_plot_plotSearch__WEBPACK_IMPORTED_MODULE_18__["PlotSearch"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 127,
      columnNumber: 13
    }
  })))), __jsx(_components_styledComponents__WEBPACK_IMPORTED_MODULE_11__["CustomRow"], {
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

/***/ }),

/***/ "./workspaces/online.ts":
/*!******************************!*\
  !*** ./workspaces/online.ts ***!
  \******************************/
/*! exports provided: summariesWorkspace, workspaces */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "summariesWorkspace", function() { return summariesWorkspace; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "workspaces", function() { return workspaces; });
var summariesWorkspace = [{
  label: 'Summary',
  foldersPath: ['Summary']
}, // {
//   label: 'Reports',
//   foldersPath: []
// },
{
  label: 'Shift',
  foldersPath: ['00 Shift']
}, {
  label: 'Info',
  foldersPath: ['Info']
}, // {
//   label: 'Certification',
//   foldersPath: []
// },
{
  label: 'Everything',
  foldersPath: []
}];
var triggerWorkspace = [{
  label: 'L1T',
  foldersPath: ['L1T']
}, {
  label: 'L1T2016EMU',
  foldersPath: ['L1T2016EMU']
}, {
  label: 'L1T2016',
  foldersPath: ['L1T2016']
}, {
  label: 'L1TEMU',
  foldersPath: ['L1TEMU']
}, {
  label: 'HLT',
  foldersPath: ['HLT']
}];
var trackerWorkspace = [{
  label: 'PixelPhase1',
  foldersPath: ['PixelPhase1']
}, {
  label: 'Pixel',
  foldersPath: ['Pixel']
}, {
  label: 'SiStrip',
  foldersPath: ['SiStrip', 'Tracking']
}];
var calorimetersWorkspace = [{
  label: 'Ecal',
  foldersPath: ['Ecal', 'EcalBarrel', 'EcalEndcap', 'EcalCalibration']
}, {
  label: 'EcalPreshower',
  foldersPath: ['EcalPreshower']
}, {
  label: 'HCAL',
  foldersPath: ['Hcal', 'Hcal2']
}, {
  label: 'HCALcalib',
  foldersPath: ['HcalCalib']
}, {
  label: 'Castor',
  foldersPath: ['Castor']
}];
var mounsWorkspace = [{
  label: 'CSC',
  foldersPath: ['CSC']
}, {
  label: 'DT',
  foldersPath: ['DT']
}, {
  label: 'RPC',
  foldersPath: ['RPC']
}];
var cttpsWorspace = [{
  label: 'TrackingStrip',
  foldersPath: ['CTPPS/TrackingStrip', 'CTPPS/common', 'CTPPS/TrackingStrip/Layouts']
}, {
  label: 'TrackingPixel',
  foldersPath: ['CTPPS/TrackingPixel', 'CTPPS/common', 'CTPPS/TrackingPixel/Layouts']
}, {
  label: 'TimingDiamond',
  foldersPath: ['CTPPS/TimingDiamond', 'CTPPS/common', 'CTPPS/TimingDiamond/Layouts']
}, {
  label: 'TimingFastSilicon',
  foldersPath: ['CTPPS/TimingFastSilicon', 'CTPPS/common', 'CTPPS/TimingFastSilicon/Layouts']
}];
var workspaces = [{
  label: 'Summaries',
  workspaces: summariesWorkspace
}, {
  label: 'Trigger',
  workspaces: triggerWorkspace
}, {
  label: 'Tracker',
  workspaces: trackerWorkspace
}, {
  label: 'Calorimeters',
  workspaces: calorimetersWorkspace
}, {
  label: 'Muons',
  workspaces: mounsWorkspace
}, {
  label: 'CTPPS',
  workspaces: cttpsWorspace
}];

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy9wbG90L3Bsb3RTZWFyY2gvaW5kZXgudHN4Iiwid2VicGFjazovL19OX0UvLi9jb21wb25lbnRzL3dvcmtzcGFjZXMvaW5kZXgudHN4Iiwid2VicGFjazovL19OX0UvLi9jb250YWluZXJzL2Rpc3BsYXkvY29udGVudC9mb2xkZXJzX2FuZF9wbG90c19jb250ZW50LnRzeCIsIndlYnBhY2s6Ly9fTl9FLy4vd29ya3NwYWNlcy9vbmxpbmUudHMiXSwibmFtZXMiOlsiUGxvdFNlYXJjaCIsInJvdXRlciIsInVzZVJvdXRlciIsInF1ZXJ5IiwiUmVhY3QiLCJwbG90X3NlYXJjaCIsInBsb3ROYW1lIiwic2V0UGxvdE5hbWUiLCJwYXJhbXMiLCJnZXRDaGFuZ2VkUXVlcnlQYXJhbXMiLCJjaGFuZ2VSb3V0ZXIiLCJlIiwidGFyZ2V0IiwidmFsdWUiLCJUYWJQYW5lIiwiVGFicyIsIldvcmtzcGFjZXMiLCJzdG9yZSIsIndvcmtzcGFjZSIsInNldFdvcmtzcGFjZSIsIndvcmtzcGFjZXMiLCJmdW5jdGlvbnNfY29uZmlnIiwibW9kZSIsIm9ubGluZVdvcmtzcGFjZSIsIm9mZmxpbmVXb3Jza3BhY2UiLCJpbml0aWFsV29ya3NwYWNlIiwibGFiZWwiLCJvcGVuV29ya3NwYWNlcyIsInRvZ2dsZVdvcmtzcGFjZXMiLCJtYXAiLCJzdWJXb3Jrc3BhY2UiLCJzZXRXb3Jrc3BhY2VUb1F1ZXJ5IiwiQ29udGVudCIsImZvbGRlcl9wYXRoIiwicnVuX251bWJlciIsImRhdGFzZXRfbmFtZSIsInVzZUNvbnRleHQiLCJ2aWV3UGxvdHNQb3NpdGlvbiIsInByb3BvcnRpb24iLCJ1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuIiwiZm9sZGVyc19wYXRoIiwibm90T2xkZXJUaGFuIiwidXNlU3RhdGUiLCJvcGVuU2V0dGluZ3MiLCJ0b2dnbGVTZXR0aW5nc01vZGFsIiwic2VsZWN0ZWRQbG90cyIsInNlbGVjdGVkX3Bsb3RzIiwidXNlRmlsdGVyRm9sZGVycyIsImZvbGRlcnNCeVBsb3RTZWFyY2giLCJwbG90cyIsImlzTG9hZGluZyIsImVycm9ycyIsInBsb3RzX3dpdGhfbGF5b3V0cyIsImZpbHRlciIsInBsb3QiLCJoYXNPd25Qcm9wZXJ0eSIsInBsb3RzX2dyb3VwZWRfYnlfbGF5b3V0cyIsImNoYWluIiwic29ydEJ5IiwiZ3JvdXBCeSIsImZpbHRlcmVkRm9sZGVycyIsImdldFNlbGVjdGVkUGxvdHMiLCJjaGFuZ2VGb2xkZXJQYXRoQnlCcmVhZGNydW1iIiwicGFyYW1ldGVycyIsInBsb3RzQXJlYVJlZiIsInVzZVJlZiIsInBsb3RzQXJlYVdpZHRoIiwic2V0UGxvdHNBcmVhV2lkdGgiLCJ1c2VFZmZlY3QiLCJjdXJyZW50IiwiY2xpZW50V2lkdGgiLCJsZW5ndGgiLCJwYWRkaW5nIiwic3VtbWFyaWVzV29ya3NwYWNlIiwiZm9sZGVyc1BhdGgiLCJ0cmlnZ2VyV29ya3NwYWNlIiwidHJhY2tlcldvcmtzcGFjZSIsImNhbG9yaW1ldGVyc1dvcmtzcGFjZSIsIm1vdW5zV29ya3NwYWNlIiwiY3R0cHNXb3JzcGFjZSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFFQTtBQUNBO0FBRUE7QUFNTyxJQUFNQSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxHQUFNO0FBQUE7O0FBQzlCLE1BQU1DLE1BQU0sR0FBR0MsNkRBQVMsRUFBeEI7QUFDQSxNQUFNQyxLQUFpQixHQUFHRixNQUFNLENBQUNFLEtBQWpDOztBQUY4Qix3QkFHRUMsOENBQUEsQ0FDOUJELEtBQUssQ0FBQ0UsV0FEd0IsQ0FIRjtBQUFBO0FBQUEsTUFHdkJDLFFBSHVCO0FBQUEsTUFHYkMsV0FIYTs7QUFPOUJILGlEQUFBLENBQWdCLFlBQU07QUFDcEIsUUFBSUQsS0FBSyxDQUFDRSxXQUFOLEtBQXNCQyxRQUExQixFQUFvQztBQUNsQyxVQUFNRSxNQUFNLEdBQUdDLHVGQUFxQixDQUFDO0FBQUVKLG1CQUFX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T0FEVDtBQUVFTixZQUFVLEVBQUUwRDtBQUZkLENBakJ3QixFQXFCeEI7QUFDRXBELE9BQUssRUFBRSxPQURUO0FBRUVOLFlBQVUsRUFBRTJEO0FBRmQsQ0FyQndCLENBQW5CIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmRjY2MxMTcyMjJhOWNjM2JhNWIzLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCBGb3JtIGZyb20gJ2FudGQvbGliL2Zvcm0vRm9ybSc7XHJcblxyXG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XHJcbmltcG9ydCB7IFN0eWxlZEZvcm1JdGVtLCBTdHlsZWRTZWFyY2ggfSBmcm9tICcuLi8uLi8uLi9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHsgUXVlcnlQcm9wcyB9IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHtcclxuICBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMsXHJcbiAgY2hhbmdlUm91dGVyLFxyXG59IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS91dGlscyc7XHJcblxyXG5cclxuZXhwb3J0IGNvbnN0IFBsb3RTZWFyY2ggPSAoKSA9PiB7XHJcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XHJcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XHJcbiAgY29uc3QgW3Bsb3ROYW1lLCBzZXRQbG90TmFtZV0gPSBSZWFjdC51c2VTdGF0ZTxzdHJpbmcgfCB1bmRlZmluZWQ+KFxyXG4gICAgcXVlcnkucGxvdF9zZWFyY2hcclxuICApO1xyXG5cclxuICBSZWFjdC51c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgaWYgKHF1ZXJ5LnBsb3Rfc2VhcmNoICE9PSBwbG90TmFtZSkge1xyXG4gICAgICBjb25zdCBwYXJhbXMgPSBnZXRDaGFuZ2VkUXVlcnlQYXJhbXMoeyBwbG90X3NlYXJjaDogcGxvdE5hbWUgfSwgcXVlcnkpO1xyXG4gICAgICBjaGFuZ2VSb3V0ZXIocGFyYW1zKTtcclxuICAgIH1cclxuICB9LCBbcGxvdE5hbWVdKTtcclxuXHJcbiAgcmV0dXJuIFJlYWN0LnVzZU1lbW8oKCkgPT4ge1xyXG4gICAgcmV0dXJuIChcclxuICAgICAgPEZvcm0gb25DaGFuZ2U9eyhlOiBhbnkpID0+IHNldFBsb3ROYW1lKGUudGFyZ2V0LnZhbHVlKX0+XHJcbiAgICAgICAgPFN0eWxlZEZvcm1JdGVtPlxyXG4gICAgICAgICAgPFN0eWxlZFNlYXJjaFxyXG4gICAgICAgICAgICBkZWZhdWx0VmFsdWU9e3F1ZXJ5LnBsb3Rfc2VhcmNofVxyXG4gICAgICAgICAgICBpZD1cInBsb3Rfc2VhcmNoXCJcclxuICAgICAgICAgICAgcGxhY2Vob2xkZXI9XCJFbnRlciBwbG90IG5hbWVcIlxyXG4gICAgICAgICAgLz5cclxuICAgICAgICA8L1N0eWxlZEZvcm1JdGVtPlxyXG4gICAgICA8L0Zvcm0+XHJcbiAgICApO1xyXG4gIH0sIFtwbG90TmFtZV0pO1xyXG59O1xyXG4iLCJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IFRhYnMsIEJ1dHRvbiB9IGZyb20gJ2FudGQnO1xyXG5cclxuaW1wb3J0IHsgd29ya3NwYWNlcyBhcyBvZmZsaW5lV29yc2twYWNlIH0gZnJvbSAnLi4vLi4vd29ya3NwYWNlcy9vZmZsaW5lJztcclxuaW1wb3J0IHsgd29ya3NwYWNlcyBhcyBvbmxpbmVXb3Jrc3BhY2UgfSBmcm9tICcuLi8uLi93b3Jrc3BhY2VzL29ubGluZSc7XHJcbmltcG9ydCB7IFN0eWxlZE1vZGFsIH0gZnJvbSAnLi4vdmlld0RldGFpbHNNZW51L3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgRm9ybSBmcm9tICdhbnRkL2xpYi9mb3JtL0Zvcm0nO1xyXG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSwgU3R5bGVkQnV0dG9uIH0gZnJvbSAnLi4vc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7IHVzZVJvdXRlciB9IGZyb20gJ25leHQvcm91dGVyJztcclxuaW1wb3J0IHsgc2V0V29ya3NwYWNlVG9RdWVyeSB9IGZyb20gJy4vdXRpbHMnO1xyXG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyB0aGVtZSB9IGZyb20gJy4uLy4uL3N0eWxlcy90aGVtZSc7XHJcbmltcG9ydCB7IGZ1bmN0aW9uc19jb25maWcgfSBmcm9tICcuLi8uLi9jb25maWcvY29uZmlnJztcclxuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnO1xyXG5cclxuY29uc3QgeyBUYWJQYW5lIH0gPSBUYWJzO1xyXG5cclxuaW50ZXJmYWNlIFdvcnNwYWNlUHJvcHMge1xyXG4gIGxhYmVsOiBzdHJpbmc7XHJcbiAgd29ya3NwYWNlczogYW55O1xyXG59XHJcbmNvbnN0IFdvcmtzcGFjZXMgPSAoKSA9PiB7XHJcbiAgY29uc3QgeyB3b3Jrc3BhY2UsIHNldFdvcmtzcGFjZSB9ID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSlcclxuXHJcbiAgY29uc3Qgd29ya3NwYWNlcyA9XHJcbiAgICBmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnID8gb25saW5lV29ya3NwYWNlIDogb2ZmbGluZVdvcnNrcGFjZTtcclxuICAgIFxyXG4gIGNvbnN0IGluaXRpYWxXb3Jrc3BhY2UgPSBmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnID8gd29ya3NwYWNlc1swXS53b3Jrc3BhY2VzWzFdLmxhYmVsIDogd29ya3NwYWNlc1swXS53b3Jrc3BhY2VzWzNdLmxhYmVsXHJcblxyXG4gIFJlYWN0LnVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBzZXRXb3Jrc3BhY2UoaW5pdGlhbFdvcmtzcGFjZSlcclxuICAgIHJldHVybiAoKSA9PiBzZXRXb3Jrc3BhY2UoaW5pdGlhbFdvcmtzcGFjZSlcclxuICB9LCBbXSlcclxuXHJcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XHJcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XHJcblxyXG4gIGNvbnN0IFtvcGVuV29ya3NwYWNlcywgdG9nZ2xlV29ya3NwYWNlc10gPSBSZWFjdC51c2VTdGF0ZShmYWxzZSk7XHJcblxyXG4gIC8vIG1ha2UgYSB3b3Jrc3BhY2Ugc2V0IGZyb20gY29udGV4dFxyXG4gIHJldHVybiAoXHJcbiAgICA8Rm9ybT5cclxuICAgICAgPFN0eWxlZEZvcm1JdGVtIGxhYmVsY29sb3I9XCJ3aGl0ZVwiIGxhYmVsPVwiV29ya3NwYWNlXCI+XHJcbiAgICAgICAgPEJ1dHRvblxyXG4gICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICB0b2dnbGVXb3Jrc3BhY2VzKCFvcGVuV29ya3NwYWNlcyk7XHJcbiAgICAgICAgICB9fVxyXG4gICAgICAgICAgdHlwZT1cImxpbmtcIlxyXG4gICAgICAgID5cclxuICAgICAgICAgIHt3b3Jrc3BhY2V9XHJcbiAgICAgICAgPC9CdXR0b24+XHJcbiAgICAgICAgPFN0eWxlZE1vZGFsXHJcbiAgICAgICAgICB0aXRsZT1cIldvcmtzcGFjZXNcIlxyXG4gICAgICAgICAgdmlzaWJsZT17b3BlbldvcmtzcGFjZXN9XHJcbiAgICAgICAgICBvbkNhbmNlbD17KCkgPT4gdG9nZ2xlV29ya3NwYWNlcyhmYWxzZSl9XHJcbiAgICAgICAgICBmb290ZXI9e251bGx9XHJcbiAgICAgICAgPlxyXG4gICAgICAgICAgPFRhYnMgZGVmYXVsdEFjdGl2ZUtleT1cIjFcIiB0eXBlPVwiY2FyZFwiPlxyXG4gICAgICAgICAgICB7d29ya3NwYWNlcy5tYXAoKHdvcmtzcGFjZTogV29yc3BhY2VQcm9wcykgPT4gKFxyXG4gICAgICAgICAgICAgIDxUYWJQYW5lIGtleT17d29ya3NwYWNlLmxhYmVsfSB0YWI9e3dvcmtzcGFjZS5sYWJlbH0+XHJcbiAgICAgICAgICAgICAgICB7d29ya3NwYWNlLndvcmtzcGFjZXMubWFwKChzdWJXb3Jrc3BhY2U6IGFueSkgPT4gKFxyXG4gICAgICAgICAgICAgICAgICA8QnV0dG9uXHJcbiAgICAgICAgICAgICAgICAgICAga2V5PXtzdWJXb3Jrc3BhY2UubGFiZWx9XHJcbiAgICAgICAgICAgICAgICAgICAgdHlwZT1cImxpbmtcIlxyXG4gICAgICAgICAgICAgICAgICAgIG9uQ2xpY2s9e2FzeW5jICgpID0+IHtcclxuICAgICAgICAgICAgICAgICAgICAgIHNldFdvcmtzcGFjZShzdWJXb3Jrc3BhY2UubGFiZWwpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgdG9nZ2xlV29ya3NwYWNlcyghb3BlbldvcmtzcGFjZXMpO1xyXG4gICAgICAgICAgICAgICAgICAgICAgLy9pZiB3b3Jrc3BhY2UgaXMgc2VsZWN0ZWQsIGZvbGRlcl9wYXRoIGluIHF1ZXJ5IGlzIHNldCB0byAnJy4gVGhlbiB3ZSBjYW4gcmVnb25pemVcclxuICAgICAgICAgICAgICAgICAgICAgIC8vdGhhdCB3b3Jrc3BhY2UgaXMgc2VsZWN0ZWQsIGFuZCB3ZWUgbmVlZCB0byBmaWx0ZXIgdGhlIGZvcnN0IGxheWVyIG9mIGZvbGRlcnMuXHJcbiAgICAgICAgICAgICAgICAgICAgICBhd2FpdCBzZXRXb3Jrc3BhY2VUb1F1ZXJ5KHF1ZXJ5LCBzdWJXb3Jrc3BhY2UubGFiZWwpO1xyXG4gICAgICAgICAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICAgICAgICB7c3ViV29ya3NwYWNlLmxhYmVsfVxyXG4gICAgICAgICAgICAgICAgICA8L0J1dHRvbj5cclxuICAgICAgICAgICAgICAgICkpfVxyXG4gICAgICAgICAgICAgIDwvVGFiUGFuZT5cclxuICAgICAgICAgICAgKSl9XHJcbiAgICAgICAgICA8L1RhYnM+XHJcbiAgICAgICAgPC9TdHlsZWRNb2RhbD5cclxuICAgICAgPC9TdHlsZWRGb3JtSXRlbT5cclxuICAgIDwvRm9ybT5cclxuICApO1xyXG59O1xyXG5cclxuZXhwb3J0IGRlZmF1bHQgV29ya3NwYWNlcztcclxuIiwiaW1wb3J0IFJlYWN0LCB7IEZDLCB1c2VTdGF0ZSwgdXNlQ29udGV4dCB9IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgQ29sLCBSb3cgfSBmcm9tICdhbnRkJztcclxuaW1wb3J0IHsgU2V0dGluZ091dGxpbmVkIH0gZnJvbSAnQGFudC1kZXNpZ24vaWNvbnMnO1xyXG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XHJcbmltcG9ydCB7IGNoYWluIH0gZnJvbSAnbG9kYXNoJztcclxuXHJcbmltcG9ydCB7IFBsb3REYXRhUHJvcHMsIFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHsgWm9vbWVkUGxvdHMgfSBmcm9tICcuLi8uLi8uLi9jb21wb25lbnRzL3Bsb3RzL3pvb21lZFBsb3RzJztcclxuaW1wb3J0IHsgVmlld0RldGFpbHNNZW51IH0gZnJvbSAnLi4vLi4vLi4vY29tcG9uZW50cy92aWV3RGV0YWlsc01lbnUnO1xyXG5pbXBvcnQgeyBEaXZXcmFwcGVyLCBab29tZWRQbG90c1dyYXBwZXIgfSBmcm9tICcuLi9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHsgRm9sZGVyUGF0aCB9IGZyb20gJy4vZm9sZGVyUGF0aCc7XHJcbmltcG9ydCB7IGNoYW5nZVJvdXRlciwgZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zLCBnZXRTZWxlY3RlZFBsb3RzIH0gZnJvbSAnLi4vdXRpbHMnO1xyXG5pbXBvcnQge1xyXG4gIEN1c3RvbVJvdyxcclxuICBTdHlsZWRTZWNvbmRhcnlCdXR0b24sXHJcbn0gZnJvbSAnLi4vLi4vLi4vY29tcG9uZW50cy9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHsgdXNlRmlsdGVyRm9sZGVycyB9IGZyb20gJy4uLy4uLy4uL2hvb2tzL3VzZUZpbHRlckZvbGRlcnMnO1xyXG5pbXBvcnQgeyBTZXR0aW5nc01vZGFsIH0gZnJvbSAnLi4vLi4vLi4vY29tcG9uZW50cy9zZXR0aW5ncyc7XHJcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vLi4vLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0JztcclxuaW1wb3J0IHsgRGlzcGxheUZvcmRlcnNPclBsb3RzIH0gZnJvbSAnLi9kaXNwbGF5X2ZvbGRlcnNfb3JfcGxvdHMnO1xyXG5pbXBvcnQgeyBVc2VmdWxMaW5rcyB9IGZyb20gJy4uLy4uLy4uL2NvbXBvbmVudHMvdXNlZnVsTGlua3MnO1xyXG5pbXBvcnQgeyBQYXJzZWRVcmxRdWVyeUlucHV0IH0gZnJvbSAncXVlcnlzdHJpbmcnO1xyXG5pbXBvcnQgV29ya3NwYWNlcyBmcm9tICcuLi8uLi8uLi9jb21wb25lbnRzL3dvcmtzcGFjZXMnO1xyXG5pbXBvcnQgeyBQbG90U2VhcmNoIH0gZnJvbSAnLi4vLi4vLi4vY29tcG9uZW50cy9wbG90cy9wbG90L3Bsb3RTZWFyY2gnO1xyXG5cclxuZXhwb3J0IGludGVyZmFjZSBQbG90SW50ZXJmYWNlIHtcclxuICBvYmo/OiBzdHJpbmc7XHJcbiAgbmFtZT86IHN0cmluZztcclxuICBwYXRoOiBzdHJpbmc7XHJcbiAgY29udGVudDogYW55O1xyXG4gIHByb3BlcnRpZXM6IGFueTtcclxuICBsYXlvdXQ/OiBzdHJpbmc7XHJcbiAgcmVwb3J0PzogYW55O1xyXG4gIHFyZXN1bHRzPzogW107XHJcbiAgcXRzdGF0dXNlcz86IFtdO1xyXG59XHJcblxyXG5leHBvcnQgaW50ZXJmYWNlIEZvbGRlclBhdGhCeUJyZWFkY3J1bWJQcm9wcyB7XHJcbiAgZm9sZGVyX3BhdGg6IHN0cmluZztcclxuICBuYW1lOiBzdHJpbmc7XHJcbn1cclxuXHJcbmludGVyZmFjZSBGb2xkZXJQcm9wcyB7XHJcbiAgZm9sZGVyX3BhdGg/OiBzdHJpbmc7XHJcbiAgcnVuX251bWJlcjogc3RyaW5nO1xyXG4gIGRhdGFzZXRfbmFtZTogc3RyaW5nO1xyXG59XHJcblxyXG5jb25zdCBDb250ZW50OiBGQzxGb2xkZXJQcm9wcz4gPSAoe1xyXG4gIGZvbGRlcl9wYXRoLFxyXG4gIHJ1bl9udW1iZXIsXHJcbiAgZGF0YXNldF9uYW1lLFxyXG59KSA9PiB7XHJcbiAgY29uc3Qge1xyXG4gICAgdmlld1Bsb3RzUG9zaXRpb24sXHJcbiAgICBwcm9wb3J0aW9uLFxyXG4gICAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbixcclxuICB9ID0gdXNlQ29udGV4dChzdG9yZSk7XHJcblxyXG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xyXG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xyXG5cclxuICBjb25zdCBwYXJhbXMgPSB7XHJcbiAgICBydW5fbnVtYmVyOiBydW5fbnVtYmVyLFxyXG4gICAgZGF0YXNldF9uYW1lOiBkYXRhc2V0X25hbWUsXHJcbiAgICBmb2xkZXJzX3BhdGg6IGZvbGRlcl9wYXRoLFxyXG4gICAgbm90T2xkZXJUaGFuOiB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuLFxyXG4gICAgcGxvdF9zZWFyY2g6IHF1ZXJ5LnBsb3Rfc2VhcmNoLFxyXG4gIH07XHJcblxyXG4gIGNvbnN0IFtvcGVuU2V0dGluZ3MsIHRvZ2dsZVNldHRpbmdzTW9kYWxdID0gdXNlU3RhdGUoZmFsc2UpO1xyXG5cclxuICBjb25zdCBzZWxlY3RlZFBsb3RzID0gcXVlcnkuc2VsZWN0ZWRfcGxvdHM7XHJcbiAgLy9maWx0ZXJpbmcgZGlyZWN0b3JpZXMgYnkgc2VsZWN0ZWQgd29ya3NwYWNlXHJcbiAgY29uc3QgeyBmb2xkZXJzQnlQbG90U2VhcmNoLCBwbG90cywgaXNMb2FkaW5nLCBlcnJvcnMgfSA9IHVzZUZpbHRlckZvbGRlcnMoXHJcbiAgICBxdWVyeSxcclxuICAgIHBhcmFtcyxcclxuICAgIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW5cclxuICApO1xyXG4gIGNvbnN0IHBsb3RzX3dpdGhfbGF5b3V0cyA9IHBsb3RzLmZpbHRlcigocGxvdCkgPT4gcGxvdC5oYXNPd25Qcm9wZXJ0eSgnbGF5b3V0JykpXHJcbiAgdmFyIHBsb3RzX2dyb3VwZWRfYnlfbGF5b3V0cyA9IGNoYWluKHBsb3RzX3dpdGhfbGF5b3V0cykuc29ydEJ5KCdsYXlvdXQnKS5ncm91cEJ5KCdsYXlvdXQnKS52YWx1ZSgpXHJcbiAgY29uc3QgZmlsdGVyZWRGb2xkZXJzOiBhbnlbXSA9IGZvbGRlcnNCeVBsb3RTZWFyY2ggPyBmb2xkZXJzQnlQbG90U2VhcmNoIDogW107XHJcbiAgY29uc3Qgc2VsZWN0ZWRfcGxvdHM6IFBsb3REYXRhUHJvcHNbXSA9IGdldFNlbGVjdGVkUGxvdHMoXHJcbiAgICBzZWxlY3RlZFBsb3RzLFxyXG4gICAgcGxvdHNcclxuICApO1xyXG5cclxuICBjb25zdCBjaGFuZ2VGb2xkZXJQYXRoQnlCcmVhZGNydW1iID0gKHBhcmFtZXRlcnM6IFBhcnNlZFVybFF1ZXJ5SW5wdXQpID0+XHJcbiAgICBjaGFuZ2VSb3V0ZXIoZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zKHBhcmFtZXRlcnMsIHF1ZXJ5KSk7XHJcblxyXG4gIGNvbnN0IHBsb3RzQXJlYVJlZiA9IFJlYWN0LnVzZVJlZjxhbnk+KG51bGwpXHJcbiAgY29uc3QgW3Bsb3RzQXJlYVdpZHRoLCBzZXRQbG90c0FyZWFXaWR0aF0gPSBSZWFjdC51c2VTdGF0ZSgwKVxyXG5cclxuICBSZWFjdC51c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgaWYgKHBsb3RzQXJlYVJlZi5jdXJyZW50KSB7XHJcbiAgICAgIHNldFBsb3RzQXJlYVdpZHRoKHBsb3RzQXJlYVJlZi5jdXJyZW50LmNsaWVudFdpZHRoKVxyXG4gICAgfVxyXG4gIH0sIFtwbG90c0FyZWFSZWYuY3VycmVudF0pXHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8PlxyXG4gICAgICA8Q3VzdG9tUm93IHNwYWNlPXsnMid9IHdpZHRoPVwiMTAwJVwiIGp1c3RpZnljb250ZW50PVwic3BhY2UtYmV0d2VlblwiPlxyXG4gICAgICAgIDxTZXR0aW5nc01vZGFsXHJcbiAgICAgICAgICBvcGVuU2V0dGluZ3M9e29wZW5TZXR0aW5nc31cclxuICAgICAgICAgIHRvZ2dsZVNldHRpbmdzTW9kYWw9e3RvZ2dsZVNldHRpbmdzTW9kYWx9XHJcbiAgICAgICAgICBpc0FueVBsb3RTZWxlY3RlZD17c2VsZWN0ZWRfcGxvdHMubGVuZ3RoID09PSAwfVxyXG4gICAgICAgIC8+XHJcbiAgICAgICAgPENvbCBzdHlsZT17eyBwYWRkaW5nOiA4IH19PlxyXG4gICAgICAgICAgPEZvbGRlclBhdGggZm9sZGVyX3BhdGg9e2ZvbGRlcl9wYXRofSBjaGFuZ2VGb2xkZXJQYXRoQnlCcmVhZGNydW1iPXtjaGFuZ2VGb2xkZXJQYXRoQnlCcmVhZGNydW1ifSAvPlxyXG4gICAgICAgIDwvQ29sPlxyXG4gICAgICAgIDxSb3cgZ3V0dGVyPXsxNn0+XHJcbiAgICAgICAgICA8Q29sPlxyXG4gICAgICAgICAgICA8VXNlZnVsTGlua3MgLz5cclxuICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgPENvbD5cclxuICAgICAgICAgICAgPFN0eWxlZFNlY29uZGFyeUJ1dHRvblxyXG4gICAgICAgICAgICAgIGljb249ezxTZXR0aW5nT3V0bGluZWQgLz59XHJcbiAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4gdG9nZ2xlU2V0dGluZ3NNb2RhbCh0cnVlKX1cclxuICAgICAgICAgICAgPlxyXG4gICAgICAgICAgICAgIFNldHRpbmdzXHJcbiAgICAgICAgICA8L1N0eWxlZFNlY29uZGFyeUJ1dHRvbj5cclxuICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgPENvbD5cclxuICAgICAgICAgICAgPFdvcmtzcGFjZXMgLz5cclxuICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgPENvbD5cclxuICAgICAgICAgICAgPFBsb3RTZWFyY2ggLz5cclxuICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgIDwvUm93PlxyXG4gICAgICA8L0N1c3RvbVJvdz5cclxuICAgICAgPEN1c3RvbVJvdyB3aWR0aD1cIjEwMCVcIj5cclxuICAgICAgICA8Vmlld0RldGFpbHNNZW51IHBsb3RzQXJlYVdpZHRoPXtwbG90c0FyZWFXaWR0aH0gc2VsZWN0ZWRfcGxvdHM9e3NlbGVjdGVkX3Bsb3RzLmxlbmd0aCA+IDB9IC8+XHJcbiAgICAgIDwvQ3VzdG9tUm93PlxyXG4gICAgICA8PlxyXG4gICAgICAgIDxEaXZXcmFwcGVyXHJcbiAgICAgICAgICBzZWxlY3RlZFBsb3RzPXtzZWxlY3RlZF9wbG90cy5sZW5ndGggPiAwfVxyXG4gICAgICAgICAgcG9zaXRpb249e3ZpZXdQbG90c1Bvc2l0aW9ufVxyXG4gICAgICAgID5cclxuICAgICAgICAgIDxEaXNwbGF5Rm9yZGVyc09yUGxvdHNcclxuICAgICAgICAgICAgcGxvdHNBcmVhUmVmPXtwbG90c0FyZWFSZWZ9XHJcbiAgICAgICAgICAgIHBsb3RzPXtwbG90c31cclxuICAgICAgICAgICAgc2VsZWN0ZWRfcGxvdHM9e3NlbGVjdGVkX3Bsb3RzfVxyXG4gICAgICAgICAgICBwbG90c19ncm91cGVkX2J5X2xheW91dHM9e3Bsb3RzX2dyb3VwZWRfYnlfbGF5b3V0c31cclxuICAgICAgICAgICAgaXNMb2FkaW5nPXtpc0xvYWRpbmd9XHJcbiAgICAgICAgICAgIHZpZXdQbG90c1Bvc2l0aW9uPXt2aWV3UGxvdHNQb3NpdGlvbn1cclxuICAgICAgICAgICAgcHJvcG9ydGlvbj17cHJvcG9ydGlvbn1cclxuICAgICAgICAgICAgZXJyb3JzPXtlcnJvcnN9XHJcbiAgICAgICAgICAgIGZpbHRlcmVkRm9sZGVycz17ZmlsdGVyZWRGb2xkZXJzfVxyXG4gICAgICAgICAgICBxdWVyeT17cXVlcnl9XHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgICAge3NlbGVjdGVkX3Bsb3RzLmxlbmd0aCA+IDAgJiYgZXJyb3JzLmxlbmd0aCA9PT0gMCAmJiAoXHJcbiAgICAgICAgICAgIDxab29tZWRQbG90c1dyYXBwZXJcclxuICAgICAgICAgICAgICBhbnlfc2VsZWN0ZWRfcGxvdHM9e3NlbGVjdGVkX3Bsb3RzLmxlbmd0aCAmJiBlcnJvcnMubGVuZ3RoID09PSAwfVxyXG4gICAgICAgICAgICAgIHByb3BvcnRpb249e3Byb3BvcnRpb259XHJcbiAgICAgICAgICAgICAgcG9zaXRpb249e3ZpZXdQbG90c1Bvc2l0aW9ufVxyXG4gICAgICAgICAgICA+XHJcbiAgICAgICAgICAgICAgPFpvb21lZFBsb3RzIHNlbGVjdGVkX3Bsb3RzPXtzZWxlY3RlZF9wbG90c30gLz5cclxuICAgICAgICAgICAgPC9ab29tZWRQbG90c1dyYXBwZXI+XHJcbiAgICAgICAgICApfVxyXG4gICAgICAgIDwvRGl2V3JhcHBlcj5cclxuICAgICAgPC8+XHJcbiAgICA8Lz5cclxuICApO1xyXG59O1xyXG5cclxuZXhwb3J0IGRlZmF1bHQgQ29udGVudDtcclxuIiwiZXhwb3J0IGludGVyZmFjZSBXb3Jza2FwYWNlc1Byb3BzIHtcclxuICBsYWJlbDogc3RyaW5nO1xyXG4gIHdvcmtzcGFjZXM6IGFueTtcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IHN1bW1hcmllc1dvcmtzcGFjZSA9IFtcclxuICB7XHJcbiAgICBsYWJlbDogJ1N1bW1hcnknLFxyXG4gICAgZm9sZGVyc1BhdGg6IFsnU3VtbWFyeSddLFxyXG4gIH0sXHJcbiAgLy8ge1xyXG4gIC8vICAgbGFiZWw6ICdSZXBvcnRzJyxcclxuICAvLyAgIGZvbGRlcnNQYXRoOiBbXVxyXG4gIC8vIH0sXHJcbiAge1xyXG4gICAgbGFiZWw6ICdTaGlmdCcsXHJcbiAgICBmb2xkZXJzUGF0aDogWycwMCBTaGlmdCddLFxyXG4gIH0sXHJcbiAge1xyXG4gICAgbGFiZWw6ICdJbmZvJyxcclxuICAgIGZvbGRlcnNQYXRoOiBbJ0luZm8nXSxcclxuICB9LFxyXG4gIC8vIHtcclxuICAvLyAgIGxhYmVsOiAnQ2VydGlmaWNhdGlvbicsXHJcbiAgLy8gICBmb2xkZXJzUGF0aDogW11cclxuICAvLyB9LFxyXG4gIHtcclxuICAgIGxhYmVsOiAnRXZlcnl0aGluZycsXHJcbiAgICBmb2xkZXJzUGF0aDogW10sXHJcbiAgfSxcclxuXTtcclxuXHJcbmNvbnN0IHRyaWdnZXJXb3Jrc3BhY2UgPSBbXHJcbiAge1xyXG4gICAgbGFiZWw6ICdMMVQnLFxyXG4gICAgZm9sZGVyc1BhdGg6IFsnTDFUJ10sXHJcbiAgfSxcclxuICB7XHJcbiAgICBsYWJlbDogJ0wxVDIwMTZFTVUnLFxyXG4gICAgZm9sZGVyc1BhdGg6IFsnTDFUMjAxNkVNVSddLFxyXG4gIH0sXHJcbiAge1xyXG4gICAgbGFiZWw6ICdMMVQyMDE2JyxcclxuICAgIGZvbGRlcnNQYXRoOiBbJ0wxVDIwMTYnXSxcclxuICB9LFxyXG4gIHtcclxuICAgIGxhYmVsOiAnTDFURU1VJyxcclxuICAgIGZvbGRlcnNQYXRoOiBbJ0wxVEVNVSddLFxyXG4gIH0sXHJcbiAge1xyXG4gICAgbGFiZWw6ICdITFQnLFxyXG4gICAgZm9sZGVyc1BhdGg6IFsnSExUJ10sXHJcbiAgfSxcclxuXTtcclxuXHJcbmNvbnN0IHRyYWNrZXJXb3Jrc3BhY2UgPSBbXHJcbiAge1xyXG4gICAgbGFiZWw6ICdQaXhlbFBoYXNlMScsXHJcbiAgICBmb2xkZXJzUGF0aDogWydQaXhlbFBoYXNlMSddLFxyXG4gIH0sXHJcbiAge1xyXG4gICAgbGFiZWw6ICdQaXhlbCcsXHJcbiAgICBmb2xkZXJzUGF0aDogWydQaXhlbCddLFxyXG4gIH0sXHJcbiAge1xyXG4gICAgbGFiZWw6ICdTaVN0cmlwJyxcclxuICAgIGZvbGRlcnNQYXRoOiBbJ1NpU3RyaXAnLCAnVHJhY2tpbmcnXSxcclxuICB9LFxyXG5dO1xyXG5cclxuY29uc3QgY2Fsb3JpbWV0ZXJzV29ya3NwYWNlID0gW1xyXG4gIHtcclxuICAgIGxhYmVsOiAnRWNhbCcsXHJcbiAgICBmb2xkZXJzUGF0aDogWydFY2FsJywgJ0VjYWxCYXJyZWwnLCAnRWNhbEVuZGNhcCcsICdFY2FsQ2FsaWJyYXRpb24nXSxcclxuICB9LFxyXG4gIHtcclxuICAgIGxhYmVsOiAnRWNhbFByZXNob3dlcicsXHJcbiAgICBmb2xkZXJzUGF0aDogWydFY2FsUHJlc2hvd2VyJ10sXHJcbiAgfSxcclxuICB7XHJcbiAgICBsYWJlbDogJ0hDQUwnLFxyXG4gICAgZm9sZGVyc1BhdGg6IFsnSGNhbCcsICdIY2FsMiddLFxyXG4gIH0sXHJcbiAge1xyXG4gICAgbGFiZWw6ICdIQ0FMY2FsaWInLFxyXG4gICAgZm9sZGVyc1BhdGg6IFsnSGNhbENhbGliJ10sXHJcbiAgfSxcclxuICB7XHJcbiAgICBsYWJlbDogJ0Nhc3RvcicsXHJcbiAgICBmb2xkZXJzUGF0aDogWydDYXN0b3InXSxcclxuICB9LFxyXG5dO1xyXG5cclxuY29uc3QgbW91bnNXb3Jrc3BhY2UgPSBbXHJcbiAge1xyXG4gICAgbGFiZWw6ICdDU0MnLFxyXG4gICAgZm9sZGVyc1BhdGg6IFsnQ1NDJ10sXHJcbiAgfSxcclxuICB7XHJcbiAgICBsYWJlbDogJ0RUJyxcclxuICAgIGZvbGRlcnNQYXRoOiBbJ0RUJ10sXHJcbiAgfSxcclxuICB7XHJcbiAgICBsYWJlbDogJ1JQQycsXHJcbiAgICBmb2xkZXJzUGF0aDogWydSUEMnXSxcclxuICB9LFxyXG5dO1xyXG5cclxuY29uc3QgY3R0cHNXb3JzcGFjZSA9IFtcclxuICB7XHJcbiAgICBsYWJlbDogJ1RyYWNraW5nU3RyaXAnLFxyXG4gICAgZm9sZGVyc1BhdGg6IFtcclxuICAgICAgJ0NUUFBTL1RyYWNraW5nU3RyaXAnLFxyXG4gICAgICAnQ1RQUFMvY29tbW9uJyxcclxuICAgICAgJ0NUUFBTL1RyYWNraW5nU3RyaXAvTGF5b3V0cycsXHJcbiAgICBdLFxyXG4gIH0sXHJcbiAge1xyXG4gICAgbGFiZWw6ICdUcmFja2luZ1BpeGVsJyxcclxuICAgIGZvbGRlcnNQYXRoOiBbXHJcbiAgICAgICdDVFBQUy9UcmFja2luZ1BpeGVsJyxcclxuICAgICAgJ0NUUFBTL2NvbW1vbicsXHJcbiAgICAgICdDVFBQUy9UcmFja2luZ1BpeGVsL0xheW91dHMnLFxyXG4gICAgXSxcclxuICB9LFxyXG4gIHtcclxuICAgIGxhYmVsOiAnVGltaW5nRGlhbW9uZCcsXHJcbiAgICBmb2xkZXJzUGF0aDogW1xyXG4gICAgICAnQ1RQUFMvVGltaW5nRGlhbW9uZCcsXHJcbiAgICAgICdDVFBQUy9jb21tb24nLFxyXG4gICAgICAnQ1RQUFMvVGltaW5nRGlhbW9uZC9MYXlvdXRzJyxcclxuICAgIF0sXHJcbiAgfSxcclxuICB7XHJcbiAgICBsYWJlbDogJ1RpbWluZ0Zhc3RTaWxpY29uJyxcclxuICAgIGZvbGRlcnNQYXRoOiBbXHJcbiAgICAgICdDVFBQUy9UaW1pbmdGYXN0U2lsaWNvbicsXHJcbiAgICAgICdDVFBQUy9jb21tb24nLFxyXG4gICAgICAnQ1RQUFMvVGltaW5nRmFzdFNpbGljb24vTGF5b3V0cycsXHJcbiAgICBdLFxyXG4gIH0sXHJcbl07XHJcblxyXG5leHBvcnQgY29uc3Qgd29ya3NwYWNlcyA9IFtcclxuICB7XHJcbiAgICBsYWJlbDogJ1N1bW1hcmllcycsXHJcbiAgICB3b3Jrc3BhY2VzOiBzdW1tYXJpZXNXb3Jrc3BhY2UsXHJcbiAgfSxcclxuICB7XHJcbiAgICBsYWJlbDogJ1RyaWdnZXInLFxyXG4gICAgd29ya3NwYWNlczogdHJpZ2dlcldvcmtzcGFjZSxcclxuICB9LFxyXG4gIHtcclxuICAgIGxhYmVsOiAnVHJhY2tlcicsXHJcbiAgICB3b3Jrc3BhY2VzOiB0cmFja2VyV29ya3NwYWNlLFxyXG4gIH0sXHJcbiAge1xyXG4gICAgbGFiZWw6ICdDYWxvcmltZXRlcnMnLFxyXG4gICAgd29ya3NwYWNlczogY2Fsb3JpbWV0ZXJzV29ya3NwYWNlLFxyXG4gIH0sXHJcbiAge1xyXG4gICAgbGFiZWw6ICdNdW9ucycsXHJcbiAgICB3b3Jrc3BhY2VzOiBtb3Vuc1dvcmtzcGFjZSxcclxuICB9LFxyXG4gIHtcclxuICAgIGxhYmVsOiAnQ1RQUFMnLFxyXG4gICAgd29ya3NwYWNlczogY3R0cHNXb3JzcGFjZSxcclxuICB9LFxyXG5dO1xyXG4iXSwic291cmNlUm9vdCI6IiJ9