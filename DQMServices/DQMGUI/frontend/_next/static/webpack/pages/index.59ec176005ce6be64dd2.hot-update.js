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

  var _React$useState3 = react__WEBPACK_IMPORTED_MODULE_2___default.a.useState(''),
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
      lineNumber: 153,
      columnNumber: 5
    }
  }, children);
};

_s(LeftSideStateProvider, "3Cy7SJKJl5v4hDr03vl72ExLRGE=");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0LnRzeCJdLCJuYW1lcyI6WyJpbml0aWFsU3RhdGUiLCJzaXplIiwic2l6ZXMiLCJtZWRpdW0iLCJub3JtYWxpemUiLCJzdGF0cyIsIm92ZXJsYXlQb3NpdGlvbiIsIm92ZXJsYXlPcHRpb25zIiwidmFsdWUiLCJvdmVybGF5IiwidW5kZWZpbmVkIiwib3ZlcmxheVBsb3RzIiwidHJpcGxlcyIsIm9wZW5PdmVybGF5RGF0YU1lbnUiLCJ2aWV3UGxvdHNQb3NpdGlvbiIsInZpZXdQb3NpdGlvbnMiLCJwcm9wb3J0aW9uIiwicGxvdHNQcm9wb3J0aW9uc09wdGlvbnMiLCJsdW1pc2VjdGlvbiIsInJpZ2h0U2lkZVNpemUiLCJmaWxsIiwiSlNST09UbW9kZSIsInNob3J0Y3V0cyIsImN1c3RvbWl6ZVByb3BzIiwieHR5cGUiLCJ4bWluIiwiTmFOIiwieG1heCIsInl0eXBlIiwieW1pbiIsInltYXgiLCJ6dHlwZSIsInptaW4iLCJ6bWF4IiwiZHJhd29wdHMiLCJ3aXRocmVmIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsIk1hdGgiLCJyb3VuZCIsIkRhdGUiLCJnZXRUaW1lIiwic3RvcmUiLCJjcmVhdGVDb250ZXh0IiwiUHJvdmlkZXIiLCJMZWZ0U2lkZVN0YXRlUHJvdmlkZXIiLCJjaGlsZHJlbiIsInVzZVN0YXRlIiwic2V0U2l6ZSIsInNldE5vcm1hbGl6ZSIsInNldFN0YXRzIiwicGxvdHNXaGljaEFyZU92ZXJsYWlkIiwic2V0UGxvdHNXaGljaEFyZU92ZXJsYWlkIiwic2V0T3ZlcmxhaVBvc2l0aW9uIiwic2V0T3ZlcmxheSIsImltYWdlUmVmU2Nyb2xsRG93biIsInNldEltYWdlUmVmU2Nyb2xsRG93biIsIlJlYWN0IiwicGxvdFNlYXJjaEZvbGRlcnMiLCJzZXRQbG90U2VhcmNoRm9sZGVycyIsIndvcmtzcGFjZSIsInNldFdvcmtzcGFjZSIsInNldFRyaXBsZXMiLCJ0b2dnbGVPdmVybGF5RGF0YU1lbnUiLCJzZXRWaWV3UGxvdHNQb3NpdGlvbiIsInNldFByb3BvcnRpb24iLCJzZXRMdW1pc2VjdGlvbiIsInNldFJpZ2h0U2lkZVNpemUiLCJzZXRKU1JPT1Rtb2RlIiwiY3VzdG9taXplIiwic2V0Q3VzdG9taXplIiwicnVuc19zZXRfZm9yX292ZXJsYXkiLCJzZXRfcnVuc19zZXRfZm9yX292ZXJsYXkiLCJ1cGRhdGUiLCJzZXRfdXBkYXRlIiwiY2hhbmdlX3ZhbHVlX2luX3JlZmVyZW5jZV90YWJsZSIsImtleSIsImlkIiwiY29weSIsImN1cnJlbnRfbGluZSIsImZpbHRlciIsImxpbmUiLCJpbmRleF9vZl9saW5lIiwiaW5kZXhPZiIsInNldF91cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUVBO0FBV0E7QUF3Qk8sSUFBTUEsWUFBaUIsR0FBRztBQUMvQkMsTUFBSSxFQUFFQywyREFBSyxDQUFDQyxNQUFOLENBQWFGLElBRFk7QUFFL0JHLFdBQVMsRUFBRSxNQUZvQjtBQUcvQkMsT0FBSyxFQUFFLElBSHdCO0FBSS9CQyxpQkFBZSxFQUFFQyxvRUFBYyxDQUFDLENBQUQsQ0FBZCxDQUFrQkMsS0FKSjtBQUsvQkMsU0FBTyxFQUFFQyxTQUxzQjtBQU0vQkMsY0FBWSxFQUFFLEVBTmlCO0FBTy9CQyxTQUFPLEVBQUUsRUFQc0I7QUFRL0JDLHFCQUFtQixFQUFFLEtBUlU7QUFTL0JDLG1CQUFpQixFQUFFQyxtRUFBYSxDQUFDLENBQUQsQ0FBYixDQUFpQlAsS0FUTDtBQVUvQlEsWUFBVSxFQUFFQyw2RUFBdUIsQ0FBQyxDQUFELENBQXZCLENBQTJCVCxLQVZSO0FBVy9CVSxhQUFXLEVBQUUsQ0FBQyxDQVhpQjtBQVkvQkMsZUFBYSxFQUFFakIsMkRBQUssQ0FBQ2tCLElBQU4sQ0FBV25CLElBWks7QUFhL0JvQixZQUFVLEVBQUUsS0FibUI7QUFjL0JDLFdBQVMsRUFBRSxFQWRvQjtB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cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgY3JlYXRlQ29udGV4dCwgdXNlU3RhdGUsIFJlYWN0RWxlbWVudCB9IGZyb20gJ3JlYWN0JztcblxuaW1wb3J0IHtcbiAgc2l6ZXMsXG4gIHZpZXdQb3NpdGlvbnMsXG4gIHBsb3RzUHJvcG9ydGlvbnNPcHRpb25zLFxufSBmcm9tICcuLi9jb21wb25lbnRzL2NvbnN0YW50cyc7XG5pbXBvcnQge1xuICBTaXplUHJvcHMsXG4gIFBsb3RQcm9wcyxcbiAgVHJpcGxlUHJvcHMsXG4gIEN1c3RvbWl6ZVByb3BzLFxufSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XG5pbXBvcnQgeyBvdmVybGF5T3B0aW9ucyB9IGZyb20gJy4uL2NvbXBvbmVudHMvY29uc3RhbnRzJztcblxuZXhwb3J0IGludGVyZmFjZSBMZWZ0U2lkZVN0YXRlUHJvdmlkZXJQcm9wcyB7XG4gIGNoaWxkcmVuOiBSZWFjdEVsZW1lbnQ7XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgTGVmdFNpZGVTdGF0ZSB7XG4gIHNpemU6IFNpemVQcm9wcztcbiAgbm9ybWFsaXplOiBib29sZWFuO1xuICBzdGF0czogYm9vbGVhbjtcbiAgb3ZlcmxheVBvc2l0aW9uOiBzdHJpbmc7XG4gIG92ZXJsYXk6IFBsb3RQcm9wc1tdO1xuICB0cmlwbGVzOiBUcmlwbGVQcm9wc1tdO1xuICBvdmVybGF5UGxvdHM6IFRyaXBsZVByb3BzW107XG4gIHdvcmtzcGFjZUZvbGRlcnM6IHN0cmluZ1tdO1xuICBvcGVuT3ZlcmxheURhdGFNZW51OiBib29sZWFuO1xuICB2aWV3UGxvdHNQb3NpdGlvbjogYm9vbGVhbjtcbiAgbHVtaXNlY3Rpb246IHN0cmluZyB8IG51bWJlcjtcbiAgcmlnaHRTaWRlU2l6ZTogU2l6ZVByb3BzO1xuICBKU1JPT1Rtb2RlOiBib29sZWFuO1xuICBjdXN0b21pemVQcm9wczogQ3VzdG9taXplUHJvcHM7XG4gIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW46IG51bWJlcjtcbn1cblxuZXhwb3J0IGNvbnN0IGluaXRpYWxTdGF0ZTogYW55ID0ge1xuICBzaXplOiBzaXplcy5tZWRpdW0uc2l6ZSxcbiAgbm9ybWFsaXplOiAnVHJ1ZScsXG4gIHN0YXRzOiB0cnVlLFxuICBvdmVybGF5UG9zaXRpb246IG92ZXJsYXlPcHRpb25zWzBdLnZhbHVlLFxuICBvdmVybGF5OiB1bmRlZmluZWQsXG4gIG92ZXJsYXlQbG90czogW10sXG4gIHRyaXBsZXM6IFtdLFxuICBvcGVuT3ZlcmxheURhdGFNZW51OiBmYWxzZSxcbiAgdmlld1Bsb3RzUG9zaXRpb246IHZpZXdQb3NpdGlvbnNbMV0udmFsdWUsXG4gIHByb3BvcnRpb246IHBsb3RzUHJvcG9ydGlvbnNPcHRpb25zWzBdLnZhbHVlLFxuICBsdW1pc2VjdGlvbjogLTEsXG4gIHJpZ2h0U2lkZVNpemU6IHNpemVzLmZpbGwuc2l6ZSxcbiAgSlNST09UbW9kZTogZmFsc2UsXG4gIHNob3J0Y3V0czogW10sXG4gIGN1c3RvbWl6ZVByb3BzOiB7XG4gICAgeHR5cGU6ICcnLFxuICAgIHhtaW46IE5hTixcbiAgICB4bWF4OiBOYU4sXG4gICAgeXR5cGU6ICcnLFxuICAgIHltaW46IE5hTixcbiAgICB5bWF4OiBOYU4sXG4gICAgenR5cGU6ICcnLFxuICAgIHptaW46IE5hTixcbiAgICB6bWF4OiBOYU4sXG4gICAgZHJhd29wdHM6ICcnLFxuICAgIHdpdGhyZWY6ICcnLFxuICB9LFxuICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuOiBNYXRoLnJvdW5kKG5ldyBEYXRlKCkuZ2V0VGltZSgpIC8gMTAwMDApICogMTAsXG59O1xuXG5leHBvcnQgaW50ZXJmYWNlIEFjdGlvblByb3BzIHtcbiAgdHlwZTogc3RyaW5nO1xuICBwYXlsb2FkOiBhbnk7XG59XG5cbmNvbnN0IHN0b3JlID0gY3JlYXRlQ29udGV4dChpbml0aWFsU3RhdGUpO1xuY29uc3QgeyBQcm92aWRlciB9ID0gc3RvcmU7XG5cbmNvbnN0IExlZnRTaWRlU3RhdGVQcm92aWRlciA9ICh7IGNoaWxkcmVuIH06IExlZnRTaWRlU3RhdGVQcm92aWRlclByb3BzKSA9PiB7XG4gIGNvbnN0IFtzaXplLCBzZXRTaXplXSA9IHVzZVN0YXRlPG51bWJlcj4oaW5pdGlhbFN0YXRlLnNpemUpO1xuICBjb25zdCBbbm9ybWFsaXplLCBzZXROb3JtYWxpemVdID0gdXNlU3RhdGU8Ym9vbGVhbj4oaW5pdGlhbFN0YXRlLm5vcm1hbGl6ZSk7XG4gIGNvbnN0IFtzdGF0cywgc2V0U3RhdHNdID0gdXNlU3RhdGU8Ym9vbGVhbj4oaW5pdGlhbFN0YXRlLnN0YXRzKTtcbiAgY29uc3QgW3Bsb3RzV2hpY2hBcmVPdmVybGFpZCwgc2V0UGxvdHNXaGljaEFyZU92ZXJsYWlkXSA9IHVzZVN0YXRlKHt9KTtcbiAgY29uc3QgW292ZXJsYXlQb3NpdGlvbiwgc2V0T3ZlcmxhaVBvc2l0aW9uXSA9IHVzZVN0YXRlKFxuICAgIGluaXRpYWxTdGF0ZS5vdmVybGF5UG9zaXRpb25cbiAgKTtcbiAgY29uc3QgW292ZXJsYXlQbG90cywgc2V0T3ZlcmxheV0gPSB1c2VTdGF0ZShpbml0aWFsU3RhdGUub3ZlcmxheVBsb3RzKTtcbiAgY29uc3QgW2ltYWdlUmVmU2Nyb2xsRG93biwgc2V0SW1hZ2VSZWZTY3JvbGxEb3duXSA9IHVzZVN0YXRlKG51bGwpO1xuICBjb25zdCBbcGxvdFNlYXJjaEZvbGRlcnMsIHNldFBsb3RTZWFyY2hGb2xkZXJzXSA9IFJlYWN0LnVzZVN0YXRlKFtdKTtcbiAgY29uc3QgW3dvcmtzcGFjZSwgc2V0V29ya3NwYWNlXSA9IFJlYWN0LnVzZVN0YXRlKCcnKTtcbiAgY29uc3QgW3RyaXBsZXMsIHNldFRyaXBsZXNdID0gUmVhY3QudXNlU3RhdGUoaW5pdGlhbFN0YXRlLnRyaXBsZXMpO1xuICBjb25zdCBbb3Blbk92ZXJsYXlEYXRhTWVudSwgdG9nZ2xlT3ZlcmxheURhdGFNZW51XSA9IFJlYWN0LnVzZVN0YXRlKFxuICAgIGluaXRpYWxTdGF0ZS5vcGVuT3ZlcmxheURhdGFNZW51XG4gICk7XG4gIGNvbnN0IFt2aWV3UGxvdHNQb3NpdGlvbiwgc2V0Vmlld1Bsb3RzUG9zaXRpb25dID0gUmVhY3QudXNlU3RhdGUoXG4gICAgaW5pdGlhbFN0YXRlLnZpZXdQbG90c1Bvc2l0aW9uXG4gICk7XG4gIGNvbnN0IFtwcm9wb3J0aW9uLCBzZXRQcm9wb3J0aW9uXSA9IFJlYWN0LnVzZVN0YXRlKGluaXRpYWxTdGF0ZS5wcm9wb3J0aW9uKTtcbiAgY29uc3QgW2x1bWlzZWN0aW9uLCBzZXRMdW1pc2VjdGlvbl0gPSBSZWFjdC51c2VTdGF0ZShcbiAgICBpbml0aWFsU3RhdGUubHVtaXNlY3Rpb25cbiAgKTtcblxuICBjb25zdCBbcmlnaHRTaWRlU2l6ZSwgc2V0UmlnaHRTaWRlU2l6ZV0gPSB1c2VTdGF0ZTxudW1iZXI+KFxuICAgIGluaXRpYWxTdGF0ZS5yaWdodFNpZGVTaXplXG4gICk7XG4gIGNvbnN0IFtKU1JPT1Rtb2RlLCBzZXRKU1JPT1Rtb2RlXSA9IHVzZVN0YXRlPGJvb2xlYW4+KGZhbHNlKTtcbiAgY29uc3QgW2N1c3RvbWl6ZSwgc2V0Q3VzdG9taXplXSA9IHVzZVN0YXRlPEN1c3RvbWl6ZVByb3BzPih7XG4gICAgeHR5cGU6ICcnLFxuICAgIHhtaW46IE5hTixcbiAgICB4bWF4OiBOYU4sXG4gICAgeXR5cGU6ICcnLFxuICAgIHltaW46IE5hTixcbiAgICB5bWF4OiBOYU4sXG4gICAgenR5cGU6ICcnLFxuICAgIHptaW46IE5hTixcbiAgICB6bWF4OiBOYU4sXG4gICAgZHJhd29wdHM6ICcnLFxuICAgIHdpdGhyZWY6ICcnLFxuICB9KTtcblxuICBjb25zdCBbcnVuc19zZXRfZm9yX292ZXJsYXksIHNldF9ydW5zX3NldF9mb3Jfb3ZlcmxheV0gPSBSZWFjdC51c2VTdGF0ZTxcbiAgICBUcmlwbGVQcm9wc1tdXG4gID4odHJpcGxlcyA/IHRyaXBsZXMgOiBbXSk7XG4gIGNvbnN0IFt1cGRhdGUsIHNldF91cGRhdGVdID0gdXNlU3RhdGU8Ym9vbGVhbj4oZmFsc2UpO1xuXG4gIGNvbnN0IGNoYW5nZV92YWx1ZV9pbl9yZWZlcmVuY2VfdGFibGUgPSAoXG4gICAgdmFsdWU6IHN0cmluZyB8IG51bWJlcixcbiAgICBrZXk6IHN0cmluZyxcbiAgICBpZDogc3RyaW5nIHwgbnVtYmVyIHwgYm9vbGVhblxuICApID0+IHtcbiAgICBjb25zdCBjb3B5ID0gWy4uLnRyaXBsZXNdO1xuICAgIC8vdHJpcGxlcyBhcmUgdGhvc2UgcnVucyB3aGljaCBhcmUgYWxyZWFkeSBvdmVybGFpZC5cbiAgICAvL3J1bnNfc2V0X2Zvcl9vdmVybGF5IGFyZSBydW5zIHdoaWNoIGFyZSBzZWtlY3RlZCBmb3Igb3ZlcmxheSxcbiAgICAvL2J1dCBub3Qgb3ZlcmxhaWQgeWV0LlxuICAgIGxldCBjdXJyZW50X2xpbmU6IFRyaXBsZVByb3BzID0gdHJpcGxlcy5maWx0ZXIoXG4gICAgICAobGluZTogVHJpcGxlUHJvcHMpID0+IGxpbmUuaWQgPT09IGlkXG4gICAgKVswXTtcbiAgICBpZiAoIWN1cnJlbnRfbGluZSkge1xuICAgICAgY3VycmVudF9saW5lID0gcnVuc19zZXRfZm9yX292ZXJsYXkuZmlsdGVyKFxuICAgICAgICAobGluZTogVHJpcGxlUHJvcHMpID0+IGxpbmUuaWQgPT09IGlkXG4gICAgICApWzBdO1xuICAgIH1cblxuICAgIGNvbnN0IGluZGV4X29mX2xpbmU6IG51bWJlciA9IGNvcHkuaW5kZXhPZihjdXJyZW50X2xpbmUpO1xuICAgIGN1cnJlbnRfbGluZVtrZXldID0gdmFsdWU7XG4gICAgY29weVtpbmRleF9vZl9saW5lXSA9IGN1cnJlbnRfbGluZTtcbiAgICBzZXRUcmlwbGVzKGNvcHkpO1xuICB9O1xuXG4gIGNvbnN0IFt1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuLCBzZXRfdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbl0gPSB1c2VTdGF0ZShcbiAgICBpbml0aWFsU3RhdGUudXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhblxuICApO1xuXG4gIHJldHVybiAoXG4gICAgPFByb3ZpZGVyXG4gICAgICB2YWx1ZT17e1xuICAgICAgICBzaXplLFxuICAgICAgICBzZXRTaXplLFxuICAgICAgICBub3JtYWxpemUsXG4gICAgICAgIHNldE5vcm1hbGl6ZSxcbiAgICAgICAgc3RhdHMsXG4gICAgICAgIHNldFN0YXRzLFxuICAgICAgICBwbG90c1doaWNoQXJlT3ZlcmxhaWQsXG4gICAgICAgIHNldFBsb3RzV2hpY2hBcmVPdmVybGFpZCxcbiAgICAgICAgb3ZlcmxheVBvc2l0aW9uLFxuICAgICAgICBzZXRPdmVybGFpUG9zaXRpb24sXG4gICAgICAgIG92ZXJsYXlQbG90cyxcbiAgICAgICAgc2V0T3ZlcmxheSxcbiAgICAgICAgaW1hZ2VSZWZTY3JvbGxEb3duLFxuICAgICAgICBzZXRJbWFnZVJlZlNjcm9sbERvd24sXG4gICAgICAgIHdvcmtzcGFjZSwgc2V0V29ya3NwYWNlLFxuICAgICAgICBwbG90U2VhcmNoRm9sZGVycyxcbiAgICAgICAgc2V0UGxvdFNlYXJjaEZvbGRlcnMsXG4gICAgICAgIGNoYW5nZV92YWx1ZV9pbl9yZWZlcmVuY2VfdGFibGUsXG4gICAgICAgIHRyaXBsZXMsXG4gICAgICAgIHNldFRyaXBsZXMsXG4gICAgICAgIG9wZW5PdmVybGF5RGF0YU1lbnUsXG4gICAgICAgIHRvZ2dsZU92ZXJsYXlEYXRhTWVudSxcbiAgICAgICAgdmlld1Bsb3RzUG9zaXRpb24sXG4gICAgICAgIHNldFZpZXdQbG90c1Bvc2l0aW9uLFxuICAgICAgICBwcm9wb3J0aW9uLFxuICAgICAgICBzZXRQcm9wb3J0aW9uLFxuICAgICAgICBsdW1pc2VjdGlvbixcbiAgICAgICAgc2V0THVtaXNlY3Rpb24sXG4gICAgICAgIHJpZ2h0U2lkZVNpemUsXG4gICAgICAgIHNldFJpZ2h0U2lkZVNpemUsXG4gICAgICAgIEpTUk9PVG1vZGUsXG4gICAgICAgIHNldEpTUk9PVG1vZGUsXG4gICAgICAgIGN1c3RvbWl6ZSxcbiAgICAgICAgc2V0Q3VzdG9taXplLFxuICAgICAgICBydW5zX3NldF9mb3Jfb3ZlcmxheSxcbiAgICAgICAgc2V0X3J1bnNfc2V0X2Zvcl9vdmVybGF5LFxuICAgICAgICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuLFxuICAgICAgICBzZXRfdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbixcbiAgICAgICAgdXBkYXRlLFxuICAgICAgICBzZXRfdXBkYXRlLFxuICAgICAgfX1cbiAgICA+XG4gICAgICB7Y2hpbGRyZW59XG4gICAgPC9Qcm92aWRlcj5cbiAgKTtcbn07XG5cbmV4cG9ydCB7IHN0b3JlLCBMZWZ0U2lkZVN0YXRlUHJvdmlkZXIgfTtcbiJdLCJzb3VyY2VSb290IjoiIn0=