webpackHotUpdate_N_E("pages/index",{

/***/ "./contexts/leftSideContext.tsx":
/*!**************************************!*\
  !*** ./contexts/leftSideContext.tsx ***!
  \**************************************/
/*! exports provided: initialState, store, LeftSideStateProvider */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "initialState", function() { return initialState; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "store", function() { return store; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "LeftSideStateProvider", function() { return LeftSideStateProvider; });
/* harmony import */ var _babel_runtime_helpers_esm_toConsumableArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/toConsumableArray */ "./node_modules/@babel/runtime/helpers/esm/toConsumableArray.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _components_constants__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../components/constants */ "./components/constants.ts");
/* harmony import */ var _workspaces_offline__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../workspaces/offline */ "./workspaces/offline.ts");



var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/contexts/leftSideContext.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_2___default.a.createElement;




var initialState = {
  size: _components_constants__WEBPACK_IMPORTED_MODULE_3__["sizes"].medium.size,
  normalize: 'True',
  stats: true,
  overlayPosition: _components_constants__WEBPACK_IMPORTED_MODULE_3__["overlayOptions"][0].value,
  overlay: undefined,
  overlayPlots: [],
  triples: [],
  openOverlayDataMenu: false,
  viewPlotsPosition: _components_constants__WEBPACK_IMPORTED_MODULE_3__["viewPositions"][1].value,
  proportion: _components_constants__WEBPACK_IMPORTED_MODULE_3__["plotsProportionsOptions"][0].value,
  lumisection: -1,
  rightSideSize: _components_constants__WEBPACK_IMPORTED_MODULE_3__["sizes"].fill.size,
  JSROOTmode: false,
  shortcuts: [],
  customizeProps: {
    xtype: '',
    xmin: NaN,
    xmax: NaN,
    ytype: '',
    ymin: NaN,
    ymax: NaN,
    ztype: '',
    zmin: NaN,
    zmax: NaN,
    drawopts: '',
    withref: ''
  },
  updated_by_not_older_than: Math.round(new Date().getTime() / 10000) * 10
};
var store = /*#__PURE__*/Object(react__WEBPACK_IMPORTED_MODULE_2__["createContext"])(initialState);
var Provider = store.Provider;

var LeftSideStateProvider = function LeftSideStateProvider(_ref) {
  _s();

  var children = _ref.children;

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initialState.size),
      size = _useState[0],
      setSize = _useState[1];

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initialState.normalize),
      normalize = _useState2[0],
      setNormalize = _useState2[1];

  var _useState3 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initialState.stats),
      stats = _useState3[0],
      setStats = _useState3[1];

  var _useState4 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])({}),
      plotsWhichAreOverlaid = _useState4[0],
      setPlotsWhichAreOverlaid = _useState4[1];

  var _useState5 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initialState.overlayPosition),
      overlayPosition = _useState5[0],
      setOverlaiPosition = _useState5[1];

  var _useState6 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initialState.overlayPlots),
      overlayPlots = _useState6[0],
      setOverlay = _useState6[1];

  var _useState7 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(null),
      imageRefScrollDown = _useState7[0],
      setImageRefScrollDown = _useState7[1];

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_2___default.a.useState([]),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState, 2),
      plotSearchFolders = _React$useState2[0],
      setPlotSearchFolders = _React$useState2[1];

  var _React$useState3 = react__WEBPACK_IMPORTED_MODULE_2___default.a.useState(_workspaces_offline__WEBPACK_IMPORTED_MODULE_4__["workspaces"][0].workspaces[1].label),
      _React$useState4 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState3, 2),
      workspace = _React$useState4[0],
      setWorkspace = _React$useState4[1];

  var _React$useState5 = react__WEBPACK_IMPORTED_MODULE_2___default.a.useState(initialState.triples),
      _React$useState6 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState5, 2),
      triples = _React$useState6[0],
      setTriples = _React$useState6[1];

  var _React$useState7 = react__WEBPACK_IMPORTED_MODULE_2___default.a.useState(initialState.openOverlayDataMenu),
      _React$useState8 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState7, 2),
      openOverlayDataMenu = _React$useState8[0],
      toggleOverlayDataMenu = _React$useState8[1];

  var _React$useState9 = react__WEBPACK_IMPORTED_MODULE_2___default.a.useState(initialState.viewPlotsPosition),
      _React$useState10 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState9, 2),
      viewPlotsPosition = _React$useState10[0],
      setViewPlotsPosition = _React$useState10[1];

  var _React$useState11 = react__WEBPACK_IMPORTED_MODULE_2___default.a.useState(initialState.proportion),
      _React$useState12 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState11, 2),
      proportion = _React$useState12[0],
      setProportion = _React$useState12[1];

  var _React$useState13 = react__WEBPACK_IMPORTED_MODULE_2___default.a.useState(initialState.lumisection),
      _React$useState14 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState13, 2),
      lumisection = _React$useState14[0],
      setLumisection = _React$useState14[1];

  var _useState8 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initialState.rightSideSize),
      rightSideSize = _useState8[0],
      setRightSideSize = _useState8[1];

  var _useState9 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(false),
      JSROOTmode = _useState9[0],
      setJSROOTmode = _useState9[1];

  var _useState10 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])({
    xtype: '',
    xmin: NaN,
    xmax: NaN,
    ytype: '',
    ymin: NaN,
    ymax: NaN,
    ztype: '',
    zmin: NaN,
    zmax: NaN,
    drawopts: '',
    withref: ''
  }),
      customize = _useState10[0],
      setCustomize = _useState10[1];

  var _React$useState15 = react__WEBPACK_IMPORTED_MODULE_2___default.a.useState(triples ? triples : []),
      _React$useState16 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState15, 2),
      runs_set_for_overlay = _React$useState16[0],
      set_runs_set_for_overlay = _React$useState16[1];

  var _useState11 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(false),
      update = _useState11[0],
      set_update = _useState11[1];

  var change_value_in_reference_table = function change_value_in_reference_table(value, key, id) {
    var copy = Object(_babel_runtime_helpers_esm_toConsumableArray__WEBPACK_IMPORTED_MODULE_0__["default"])(triples); //triples are those runs which are already overlaid.
    //runs_set_for_overlay are runs which are sekected for overlay,
    //but not overlaid yet.


    var current_line = triples.filter(function (line) {
      return line.id === id;
    })[0];

    if (!current_line) {
      current_line = runs_set_for_overlay.filter(function (line) {
        return line.id === id;
      })[0];
    }

    var index_of_line = copy.indexOf(current_line);
    current_line[key] = value;
    copy[index_of_line] = current_line;
    setTriples(copy);
  };

  var _useState12 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initialState.updated_by_not_older_than),
      updated_by_not_older_than = _useState12[0],
      set_updated_by_not_older_than = _useState12[1];

  return __jsx(Provider, {
    value: {
      size: size,
      setSize: setSize,
      normalize: normalize,
      setNormalize: setNormalize,
      stats: stats,
      setStats: setStats,
      plotsWhichAreOverlaid: plotsWhichAreOverlaid,
      setPlotsWhichAreOverlaid: setPlotsWhichAreOverlaid,
      overlayPosition: overlayPosition,
      setOverlaiPosition: setOverlaiPosition,
      overlayPlots: overlayPlots,
      setOverlay: setOverlay,
      imageRefScrollDown: imageRefScrollDown,
      setImageRefScrollDown: setImageRefScrollDown,
      workspace: workspace,
      setWorkspace: setWorkspace,
      plotSearchFolders: plotSearchFolders,
      setPlotSearchFolders: setPlotSearchFolders,
      change_value_in_reference_table: change_value_in_reference_table,
      triples: triples,
      setTriples: setTriples,
      openOverlayDataMenu: openOverlayDataMenu,
      toggleOverlayDataMenu: toggleOverlayDataMenu,
      viewPlotsPosition: viewPlotsPosition,
      setViewPlotsPosition: setViewPlotsPosition,
      proportion: proportion,
      setProportion: setProportion,
      lumisection: lumisection,
      setLumisection: setLumisection,
      rightSideSize: rightSideSize,
      setRightSideSize: setRightSideSize,
      JSROOTmode: JSROOTmode,
      setJSROOTmode: setJSROOTmode,
      customize: customize,
      setCustomize: setCustomize,
      runs_set_for_overlay: runs_set_for_overlay,
      set_runs_set_for_overlay: set_runs_set_for_overlay,
      updated_by_not_older_than: updated_by_not_older_than,
      set_updated_by_not_older_than: set_updated_by_not_older_than,
      update: update,
      set_update: set_update
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 155,
      columnNumber: 5
    }
  }, children);
};

_s(LeftSideStateProvider, "vUdwKCdznS5V5G1C87Fvew5Yl7Y=");

_c = LeftSideStateProvider;


var _c;

$RefreshReg$(_c, "LeftSideStateProvider");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0LnRzeCJdLCJuYW1lcyI6WyJpbml0aWFsU3RhdGUiLCJzaXplIiwic2l6ZXMiLCJtZWRpdW0iLCJub3JtYWxpemUiLCJzdGF0cyIsIm92ZXJsYXlQb3NpdGlvbiIsIm92ZXJsYXlPcHRpb25zIiwidmFsdWUiLCJvdmVybGF5IiwidW5kZWZpbmVkIiwib3ZlcmxheVBsb3RzIiwidHJpcGxlcyIsIm9wZW5PdmVybGF5RGF0YU1lbnUiLCJ2aWV3UGxvdHNQb3NpdGlvbiIsInZpZXdQb3NpdGlvbnMiLCJwcm9wb3J0aW9uIiwicGxvdHNQcm9wb3J0aW9uc09wdGlvbnMiLCJsdW1pc2VjdGlvbiIsInJpZ2h0U2lkZVNpemUiLCJmaWxsIiwiSlNST09UbW9kZSIsInNob3J0Y3V0cyIsImN1c3RvbWl6ZVByb3BzIiwieHR5cGUiLCJ4bWluIiwiTmFOIiwieG1heCIsInl0eXBlIiwieW1pbiIsInltYXgiLCJ6dHlwZSIsInptaW4iLCJ6bWF4IiwiZHJhd29wdHMiLCJ3aXRocmVmIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsIk1hdGgiLCJyb3VuZCIsIkRhdGUiLCJnZXRUaW1lIiwic3RvcmUiLCJjcmVhdGVDb250ZXh0IiwiUHJvdmlkZXIiLCJMZWZ0U2lkZVN0YXRlUHJvdmlkZXIiLCJjaGlsZHJlbiIsInVzZVN0YXRlIiwic2V0U2l6ZSIsInNldE5vcm1hbGl6ZSIsInNldFN0YXRzIiwicGxvdHNXaGljaEFyZU92ZXJsYWlkIiwic2V0UGxvdHNXaGljaEFyZU92ZXJsYWlkIiwic2V0T3ZlcmxhaVBvc2l0aW9uIiwic2V0T3ZlcmxheSIsImltYWdlUmVmU2Nyb2xsRG93biIsInNldEltYWdlUmVmU2Nyb2xsRG93biIsIlJlYWN0IiwicGxvdFNlYXJjaEZvbGRlcnMiLCJzZXRQbG90U2VhcmNoRm9sZGVycyIsIndvcmtzcGFjZXMiLCJsYWJlbCIsIndvcmtzcGFjZSIsInNldFdvcmtzcGFjZSIsInNldFRyaXBsZXMiLCJ0b2dnbGVPdmVybGF5RGF0YU1lbnUiLCJzZXRWaWV3UGxvdHNQb3NpdGlvbiIsInNldFByb3BvcnRpb24iLCJzZXRMdW1pc2VjdGlvbiIsInNldFJpZ2h0U2lkZVNpemUiLCJzZXRKU1JPT1Rtb2RlIiwiY3VzdG9taXplIiwic2V0Q3VzdG9taXplIiwicnVuc19zZXRfZm9yX292ZXJsYXkiLCJzZXRfcnVuc19zZXRfZm9yX292ZXJsYXkiLCJ1cGRhdGUiLCJzZXRfdXBkYXRlIiwiY2hhbmdlX3ZhbHVlX2luX3JlZmVyZW5jZV90YWJsZSIsImtleSIsImlkIiwiY29weSIsImN1cnJlbnRfbGluZSIsImZpbHRlciIsImxpbmUiLCJpbmRleF9vZl9saW5lIiwiaW5kZXhPZiIsInNldF91cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFHQTtBQVdBO0FBQ0E7QUF3Qk8sSUFBTUEsWUFBaUIsR0FBRztBQUMvQkMsTUFBSSxFQUFFQywyREFBSyxDQUFDQyxNQUFOLENBQWFGLElBRFk7QUFFL0JHLFdBQVMsRUFBRSxNQUZvQjtBQUcvQkMsT0FBSyxFQUFFLElBSHdCO0FBSS9CQyxpQkFBZSxFQUFFQyxvRUFBYyxDQUFDLENBQUQsQ0FBZCxDQUFrQkMsS0FKSjtBQUsvQkMsU0FBTyxFQUFFQyxTQUxzQjtBQU0vQkMsY0FBWSxFQUFFLEVBTmlCO0FBTy9CQyxTQUFPLEVBQUUsRUFQc0I7QUFRL0JDLHFCQUFtQixFQUFFLEtBUlU7QUFTL0JDLG1CQUFpQixFQUFFQyxtRUFBYSxDQUFDLENBQUQsQ0FBYixDQUFpQlAsS0FUTDtBQVUvQlEsWUFBVSxFQUFFQyw2RUFBdUIsQ0FBQyxDQUFELENBQXZCLENBQTJCVCxLQVZSO0FBVy9CVSxhQUFXLEVBQUUsQ0FBQyxDQVhpQjtBQVkvQkMsZUFBYSxFQUFFakIsMkRBQUssQ0FBQ2tCLElBQU4sQ0FBV25CLElBWks7QUFhL0JvQixZQUFVLEVBQUUsS0Fi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Qjs7S0FBQUEscUI7QUE2SE4iLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNGFmZDJlNWQyODdlNjEwZGUwMDQuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCwgeyBjcmVhdGVDb250ZXh0LCB1c2VTdGF0ZSwgUmVhY3RFbGVtZW50IH0gZnJvbSAncmVhY3QnO1xuaW1wb3J0IHsgdjQgYXMgdXVpZHY0IH0gZnJvbSAndXVpZCc7XG5cbmltcG9ydCB7XG4gIHNpemVzLFxuICB2aWV3UG9zaXRpb25zLFxuICBwbG90c1Byb3BvcnRpb25zT3B0aW9ucyxcbn0gZnJvbSAnLi4vY29tcG9uZW50cy9jb25zdGFudHMnO1xuaW1wb3J0IHtcbiAgU2l6ZVByb3BzLFxuICBQbG90UHJvcHMsXG4gIFRyaXBsZVByb3BzLFxuICBDdXN0b21pemVQcm9wcyxcbn0gZnJvbSAnLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgb3ZlcmxheU9wdGlvbnMgfSBmcm9tICcuLi9jb21wb25lbnRzL2NvbnN0YW50cyc7XG5pbXBvcnQgeyB3b3Jrc3BhY2VzIH0gZnJvbSAnLi4vd29ya3NwYWNlcy9vZmZsaW5lJztcblxuZXhwb3J0IGludGVyZmFjZSBMZWZ0U2lkZVN0YXRlUHJvdmlkZXJQcm9wcyB7XG4gIGNoaWxkcmVuOiBSZWFjdEVsZW1lbnQ7XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgTGVmdFNpZGVTdGF0ZSB7XG4gIHNpemU6IFNpemVQcm9wcztcbiAgbm9ybWFsaXplOiBib29sZWFuO1xuICBzdGF0czogYm9vbGVhbjtcbiAgb3ZlcmxheVBvc2l0aW9uOiBzdHJpbmc7XG4gIG92ZXJsYXk6IFBsb3RQcm9wc1tdO1xuICB0cmlwbGVzOiBUcmlwbGVQcm9wc1tdO1xuICBvdmVybGF5UGxvdHM6IFRyaXBsZVByb3BzW107XG4gIHdvcmtzcGFjZUZvbGRlcnM6IHN0cmluZ1tdO1xuICBvcGVuT3ZlcmxheURhdGFNZW51OiBib29sZWFuO1xuICB2aWV3UGxvdHNQb3NpdGlvbjogYm9vbGVhbjtcbiAgbHVtaXNlY3Rpb246IHN0cmluZyB8IG51bWJlcjtcbiAgcmlnaHRTaWRlU2l6ZTogU2l6ZVByb3BzO1xuICBKU1JPT1Rtb2RlOiBib29sZWFuO1xuICBjdXN0b21pemVQcm9wczogQ3VzdG9taXplUHJvcHM7XG4gIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW46IG51bWJlcjtcbn1cblxuZXhwb3J0IGNvbnN0IGluaXRpYWxTdGF0ZTogYW55ID0ge1xuICBzaXplOiBzaXplcy5tZWRpdW0uc2l6ZSxcbiAgbm9ybWFsaXplOiAnVHJ1ZScsXG4gIHN0YXRzOiB0cnVlLFxuICBvdmVybGF5UG9zaXRpb246IG92ZXJsYXlPcHRpb25zWzBdLnZhbHVlLFxuICBvdmVybGF5OiB1bmRlZmluZWQsXG4gIG92ZXJsYXlQbG90czogW10sXG4gIHRyaXBsZXM6IFtdLFxuICBvcGVuT3ZlcmxheURhdGFNZW51OiBmYWxzZSxcbiAgdmlld1Bsb3RzUG9zaXRpb246IHZpZXdQb3NpdGlvbnNbMV0udmFsdWUsXG4gIHByb3BvcnRpb246IHBsb3RzUHJvcG9ydGlvbnNPcHRpb25zWzBdLnZhbHVlLFxuICBsdW1pc2VjdGlvbjogLTEsXG4gIHJpZ2h0U2lkZVNpemU6IHNpemVzLmZpbGwuc2l6ZSxcbiAgSlNST09UbW9kZTogZmFsc2UsXG4gIHNob3J0Y3V0czogW10sXG4gIGN1c3RvbWl6ZVByb3BzOiB7XG4gICAgeHR5cGU6ICcnLFxuICAgIHhtaW46IE5hTixcbiAgICB4bWF4OiBOYU4sXG4gICAgeXR5cGU6ICcnLFxuICAgIHltaW46IE5hTixcbiAgICB5bWF4OiBOYU4sXG4gICAgenR5cGU6ICcnLFxuICAgIHptaW46IE5hTixcbiAgICB6bWF4OiBOYU4sXG4gICAgZHJhd29wdHM6ICcnLFxuICAgIHdpdGhyZWY6ICcnLFxuICB9LFxuICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuOiBNYXRoLnJvdW5kKG5ldyBEYXRlKCkuZ2V0VGltZSgpIC8gMTAwMDApICogMTAsXG59O1xuXG5leHBvcnQgaW50ZXJmYWNlIEFjdGlvblByb3BzIHtcbiAgdHlwZTogc3RyaW5nO1xuICBwYXlsb2FkOiBhbnk7XG59XG5cbmNvbnN0IHN0b3JlID0gY3JlYXRlQ29udGV4dChpbml0aWFsU3RhdGUpO1xuY29uc3QgeyBQcm92aWRlciB9ID0gc3RvcmU7XG5cbmNvbnN0IExlZnRTaWRlU3RhdGVQcm92aWRlciA9ICh7IGNoaWxkcmVuIH06IExlZnRTaWRlU3RhdGVQcm92aWRlclByb3BzKSA9PiB7XG4gIGNvbnN0IFtzaXplLCBzZXRTaXplXSA9IHVzZVN0YXRlPG51bWJlcj4oaW5pdGlhbFN0YXRlLnNpemUpO1xuICBjb25zdCBbbm9ybWFsaXplLCBzZXROb3JtYWxpemVdID0gdXNlU3RhdGU8Ym9vbGVhbj4oaW5pdGlhbFN0YXRlLm5vcm1hbGl6ZSk7XG4gIGNvbnN0IFtzdGF0cywgc2V0U3RhdHNdID0gdXNlU3RhdGU8Ym9vbGVhbj4oaW5pdGlhbFN0YXRlLnN0YXRzKTtcbiAgY29uc3QgW3Bsb3RzV2hpY2hBcmVPdmVybGFpZCwgc2V0UGxvdHNXaGljaEFyZU92ZXJsYWlkXSA9IHVzZVN0YXRlKHt9KTtcbiAgY29uc3QgW292ZXJsYXlQb3NpdGlvbiwgc2V0T3ZlcmxhaVBvc2l0aW9uXSA9IHVzZVN0YXRlKFxuICAgIGluaXRpYWxTdGF0ZS5vdmVybGF5UG9zaXRpb25cbiAgKTtcbiAgY29uc3QgW292ZXJsYXlQbG90cywgc2V0T3ZlcmxheV0gPSB1c2VTdGF0ZShpbml0aWFsU3RhdGUub3ZlcmxheVBsb3RzKTtcbiAgY29uc3QgW2ltYWdlUmVmU2Nyb2xsRG93biwgc2V0SW1hZ2VSZWZTY3JvbGxEb3duXSA9IHVzZVN0YXRlKG51bGwpO1xuICBjb25zdCBbcGxvdFNlYXJjaEZvbGRlcnMsIHNldFBsb3RTZWFyY2hGb2xkZXJzXSA9IFJlYWN0LnVzZVN0YXRlKFtdKTtcbiAgY29uc3QgW3dvcmtzcGFjZSwgc2V0V29ya3NwYWNlXSA9IFJlYWN0LnVzZVN0YXRlKHdvcmtzcGFjZXNbMF0ud29ya3NwYWNlc1sxXS5sYWJlbCk7XG4gIGNvbnN0IFt0cmlwbGVzLCBzZXRUcmlwbGVzXSA9IFJlYWN0LnVzZVN0YXRlKGluaXRpYWxTdGF0ZS50cmlwbGVzKTtcbiAgY29uc3QgW29wZW5PdmVybGF5RGF0YU1lbnUsIHRvZ2dsZU92ZXJsYXlEYXRhTWVudV0gPSBSZWFjdC51c2VTdGF0ZShcbiAgICBpbml0aWFsU3RhdGUub3Blbk92ZXJsYXlEYXRhTWVudVxuICApO1xuICBjb25zdCBbdmlld1Bsb3RzUG9zaXRpb24sIHNldFZpZXdQbG90c1Bvc2l0aW9uXSA9IFJlYWN0LnVzZVN0YXRlKFxuICAgIGluaXRpYWxTdGF0ZS52aWV3UGxvdHNQb3NpdGlvblxuICApO1xuICBjb25zdCBbcHJvcG9ydGlvbiwgc2V0UHJvcG9ydGlvbl0gPSBSZWFjdC51c2VTdGF0ZShpbml0aWFsU3RhdGUucHJvcG9ydGlvbik7XG4gIGNvbnN0IFtsdW1pc2VjdGlvbiwgc2V0THVtaXNlY3Rpb25dID0gUmVhY3QudXNlU3RhdGUoXG4gICAgaW5pdGlhbFN0YXRlLmx1bWlzZWN0aW9uXG4gICk7XG5cbiAgY29uc3QgW3JpZ2h0U2lkZVNpemUsIHNldFJpZ2h0U2lkZVNpemVdID0gdXNlU3RhdGU8bnVtYmVyPihcbiAgICBpbml0aWFsU3RhdGUucmlnaHRTaWRlU2l6ZVxuICApO1xuICBjb25zdCBbSlNST09UbW9kZSwgc2V0SlNST09UbW9kZV0gPSB1c2VTdGF0ZTxib29sZWFuPihmYWxzZSk7XG4gIGNvbnN0IFtjdXN0b21pemUsIHNldEN1c3RvbWl6ZV0gPSB1c2VTdGF0ZTxDdXN0b21pemVQcm9wcz4oe1xuICAgIHh0eXBlOiAnJyxcbiAgICB4bWluOiBOYU4sXG4gICAgeG1heDogTmFOLFxuICAgIHl0eXBlOiAnJyxcbiAgICB5bWluOiBOYU4sXG4gICAgeW1heDogTmFOLFxuICAgIHp0eXBlOiAnJyxcbiAgICB6bWluOiBOYU4sXG4gICAgem1heDogTmFOLFxuICAgIGRyYXdvcHRzOiAnJyxcbiAgICB3aXRocmVmOiAnJyxcbiAgfSk7XG5cbiAgY29uc3QgW3J1bnNfc2V0X2Zvcl9vdmVybGF5LCBzZXRfcnVuc19zZXRfZm9yX292ZXJsYXldID0gUmVhY3QudXNlU3RhdGU8XG4gICAgVHJpcGxlUHJvcHNbXVxuICA+KHRyaXBsZXMgPyB0cmlwbGVzIDogW10pO1xuICBjb25zdCBbdXBkYXRlLCBzZXRfdXBkYXRlXSA9IHVzZVN0YXRlPGJvb2xlYW4+KGZhbHNlKTtcblxuICBjb25zdCBjaGFuZ2VfdmFsdWVfaW5fcmVmZXJlbmNlX3RhYmxlID0gKFxuICAgIHZhbHVlOiBzdHJpbmcgfCBudW1iZXIsXG4gICAga2V5OiBzdHJpbmcsXG4gICAgaWQ6IHN0cmluZyB8IG51bWJlciB8IGJvb2xlYW5cbiAgKSA9PiB7XG4gICAgY29uc3QgY29weSA9IFsuLi50cmlwbGVzXTtcbiAgICAvL3RyaXBsZXMgYXJlIHRob3NlIHJ1bnMgd2hpY2ggYXJlIGFscmVhZHkgb3ZlcmxhaWQuXG4gICAgLy9ydW5zX3NldF9mb3Jfb3ZlcmxheSBhcmUgcnVucyB3aGljaCBhcmUgc2VrZWN0ZWQgZm9yIG92ZXJsYXksXG4gICAgLy9idXQgbm90IG92ZXJsYWlkIHlldC5cbiAgICBsZXQgY3VycmVudF9saW5lOiBUcmlwbGVQcm9wcyA9IHRyaXBsZXMuZmlsdGVyKFxuICAgICAgKGxpbmU6IFRyaXBsZVByb3BzKSA9PiBsaW5lLmlkID09PSBpZFxuICAgIClbMF07XG4gICAgaWYgKCFjdXJyZW50X2xpbmUpIHtcbiAgICAgIGN1cnJlbnRfbGluZSA9IHJ1bnNfc2V0X2Zvcl9vdmVybGF5LmZpbHRlcihcbiAgICAgICAgKGxpbmU6IFRyaXBsZVByb3BzKSA9PiBsaW5lLmlkID09PSBpZFxuICAgICAgKVswXTtcbiAgICB9XG5cbiAgICBjb25zdCBpbmRleF9vZl9saW5lOiBudW1iZXIgPSBjb3B5LmluZGV4T2YoY3VycmVudF9saW5lKTtcbiAgICBjdXJyZW50X2xpbmVba2V5XSA9IHZhbHVlO1xuICAgIGNvcHlbaW5kZXhfb2ZfbGluZV0gPSBjdXJyZW50X2xpbmU7XG4gICAgc2V0VHJpcGxlcyhjb3B5KTtcbiAgfTtcblxuICBjb25zdCBbdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiwgc2V0X3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW5dID0gdXNlU3RhdGUoXG4gICAgaW5pdGlhbFN0YXRlLnVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW5cbiAgKTtcblxuICByZXR1cm4gKFxuICAgIDxQcm92aWRlclxuICAgICAgdmFsdWU9e3tcbiAgICAgICAgc2l6ZSxcbiAgICAgICAgc2V0U2l6ZSxcbiAgICAgICAgbm9ybWFsaXplLFxuICAgICAgICBzZXROb3JtYWxpemUsXG4gICAgICAgIHN0YXRzLFxuICAgICAgICBzZXRTdGF0cyxcbiAgICAgICAgcGxvdHNXaGljaEFyZU92ZXJsYWlkLFxuICAgICAgICBzZXRQbG90c1doaWNoQXJlT3ZlcmxhaWQsXG4gICAgICAgIG92ZXJsYXlQb3NpdGlvbixcbiAgICAgICAgc2V0T3ZlcmxhaVBvc2l0aW9uLFxuICAgICAgICBvdmVybGF5UGxvdHMsXG4gICAgICAgIHNldE92ZXJsYXksXG4gICAgICAgIGltYWdlUmVmU2Nyb2xsRG93bixcbiAgICAgICAgc2V0SW1hZ2VSZWZTY3JvbGxEb3duLFxuICAgICAgICB3b3Jrc3BhY2UsIHNldFdvcmtzcGFjZSxcbiAgICAgICAgcGxvdFNlYXJjaEZvbGRlcnMsXG4gICAgICAgIHNldFBsb3RTZWFyY2hGb2xkZXJzLFxuICAgICAgICBjaGFuZ2VfdmFsdWVfaW5fcmVmZXJlbmNlX3RhYmxlLFxuICAgICAgICB0cmlwbGVzLFxuICAgICAgICBzZXRUcmlwbGVzLFxuICAgICAgICBvcGVuT3ZlcmxheURhdGFNZW51LFxuICAgICAgICB0b2dnbGVPdmVybGF5RGF0YU1lbnUsXG4gICAgICAgIHZpZXdQbG90c1Bvc2l0aW9uLFxuICAgICAgICBzZXRWaWV3UGxvdHNQb3NpdGlvbixcbiAgICAgICAgcHJvcG9ydGlvbixcbiAgICAgICAgc2V0UHJvcG9ydGlvbixcbiAgICAgICAgbHVtaXNlY3Rpb24sXG4gICAgICAgIHNldEx1bWlzZWN0aW9uLFxuICAgICAgICByaWdodFNpZGVTaXplLFxuICAgICAgICBzZXRSaWdodFNpZGVTaXplLFxuICAgICAgICBKU1JPT1Rtb2RlLFxuICAgICAgICBzZXRKU1JPT1Rtb2RlLFxuICAgICAgICBjdXN0b21pemUsXG4gICAgICAgIHNldEN1c3RvbWl6ZSxcbiAgICAgICAgcnVuc19zZXRfZm9yX292ZXJsYXksXG4gICAgICAgIHNldF9ydW5zX3NldF9mb3Jfb3ZlcmxheSxcbiAgICAgICAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbixcbiAgICAgICAgc2V0X3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4sXG4gICAgICAgIHVwZGF0ZSxcbiAgICAgICAgc2V0X3VwZGF0ZSxcbiAgICAgIH19XG4gICAgPlxuICAgICAge2NoaWxkcmVufVxuICAgIDwvUHJvdmlkZXI+XG4gICk7XG59O1xuXG5leHBvcnQgeyBzdG9yZSwgTGVmdFNpZGVTdGF0ZVByb3ZpZGVyIH07XG4iXSwic291cmNlUm9vdCI6IiJ9