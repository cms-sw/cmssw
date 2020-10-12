webpackHotUpdate_N_E("pages/_app",{

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

  var _React$useState3 = react__WEBPACK_IMPORTED_MODULE_2___default.a.useState([]),
      _React$useState4 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_React$useState3, 2),
      workspaceFolders = _React$useState4[0],
      setWorkspaceFolders = _React$useState4[1];

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
      workspaceFolders: workspaceFolders,
      setWorkspaceFolders: setWorkspaceFolders,
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
      lineNumber: 154,
      columnNumber: 5
    }
  }, children);
};

_s(LeftSideStateProvider, "xcLx0ajuCJaEiSXGsBojwAKaknA=");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0LnRzeCJdLCJuYW1lcyI6WyJpbml0aWFsU3RhdGUiLCJzaXplIiwic2l6ZXMiLCJtZWRpdW0iLCJub3JtYWxpemUiLCJzdGF0cyIsIm92ZXJsYXlQb3NpdGlvbiIsIm92ZXJsYXlPcHRpb25zIiwidmFsdWUiLCJvdmVybGF5IiwidW5kZWZpbmVkIiwib3ZlcmxheVBsb3RzIiwidHJpcGxlcyIsIm9wZW5PdmVybGF5RGF0YU1lbnUiLCJ2aWV3UGxvdHNQb3NpdGlvbiIsInZpZXdQb3NpdGlvbnMiLCJwcm9wb3J0aW9uIiwicGxvdHNQcm9wb3J0aW9uc09wdGlvbnMiLCJsdW1pc2VjdGlvbiIsInJpZ2h0U2lkZVNpemUiLCJmaWxsIiwiSlNST09UbW9kZSIsInNob3J0Y3V0cyIsImN1c3RvbWl6ZVByb3BzIiwieHR5cGUiLCJ4bWluIiwiTmFOIiwieG1heCIsInl0eXBlIiwieW1pbiIsInltYXgiLCJ6dHlwZSIsInptaW4iLCJ6bWF4IiwiZHJhd29wdHMiLCJ3aXRocmVmIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsIk1hdGgiLCJyb3VuZCIsIkRhdGUiLCJnZXRUaW1lIiwic3RvcmUiLCJjcmVhdGVDb250ZXh0IiwiUHJvdmlkZXIiLCJMZWZ0U2lkZVN0YXRlUHJvdmlkZXIiLCJjaGlsZHJlbiIsInVzZVN0YXRlIiwic2V0U2l6ZSIsInNldE5vcm1hbGl6ZSIsInNldFN0YXRzIiwicGxvdHNXaGljaEFyZU92ZXJsYWlkIiwic2V0UGxvdHNXaGljaEFyZU92ZXJsYWlkIiwic2V0T3ZlcmxhaVBvc2l0aW9uIiwic2V0T3ZlcmxheSIsImltYWdlUmVmU2Nyb2xsRG93biIsInNldEltYWdlUmVmU2Nyb2xsRG93biIsIlJlYWN0IiwicGxvdFNlYXJjaEZvbGRlcnMiLCJzZXRQbG90U2VhcmNoRm9sZGVycyIsIndvcmtzcGFjZUZvbGRlcnMiLCJzZXRXb3Jrc3BhY2VGb2xkZXJzIiwic2V0VHJpcGxlcyIsInRvZ2dsZU92ZXJsYXlEYXRhTWVudSIsInNldFZpZXdQbG90c1Bvc2l0aW9uIiwic2V0UHJvcG9ydGlvbiIsInNldEx1bWlzZWN0aW9uIiwic2V0UmlnaHRTaWRlU2l6ZSIsInNldEpTUk9PVG1vZGUiLCJjdXN0b21pemUiLCJzZXRDdXN0b21pemUiLCJydW5zX3NldF9mb3Jfb3ZlcmxheSIsInNldF9ydW5zX3NldF9mb3Jfb3ZlcmxheSIsInVwZGF0ZSIsInNldF91cGRhdGUiLCJjaGFuZ2VfdmFsdWVfaW5fcmVmZXJlbmNlX3RhYmxlIiwia2V5IiwiaWQiLCJjb3B5IiwiY3VycmVudF9saW5lIiwiZmlsdGVyIiwibGluZSIsImluZGV4X29mX2xpbmUiLCJpbmRleE9mIiwic2V0X3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4iXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBR0E7QUFXQTtBQXdCTyxJQUFNQSxZQUFpQixHQUFHO0FBQy9CQyxNQUFJLEVBQUVDLDJEQUFLLENBQUNDLE1BQU4sQ0FBYUYsSUFEWTtBQUUvQkcsV0FBUyxFQUFFLE1BRm9CO0FBRy9CQyxPQUFLLEVBQUUsSUFId0I7QUFJL0JDLGlCQUFlLEVBQUVDLG9FQUFjLENBQUMsQ0FBRCxDQUFkLENBQWtCQyxLQUpKO0FBSy9CQyxTQUFPLEVBQUVDLFNBTHNCO0FBTS9CQyxjQUFZLEVBQUUsRUFOaUI7QUFPL0JDLFNBQU8sRUFBRSxFQVBzQjtBQVEvQkMscUJBQW1CLEVBQUUsS0FSVTtBQVMvQkMsbUJBQWlCLEVBQUVDLG1FQUFhLENBQUMsQ0FBRCxDQUFiLENBQWlCUCxLQVRMO0FBVS9CUSxZQUFVLEVBQUVDLDZFQUF1QixDQUFDLENBQUQsQ0FBdkIsQ0FBMkJULEtBVlI7QUFXL0JVLGFBQVcsRUFBRSxDQUFDLENBWGlCO0FBWS9CQyxlQUFhLEVBQUVqQiwyREFBSyxDQUFDa0IsSUFBTixDQUFXbkIsSUFaSztBQWEvQm9CLFlBQVUsRUFBRSxLQWJtQjtBQWMvQkMsV0FB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ODQ5ZWRiMTM0YTY2NjE4ZjBlLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgY3JlYXRlQ29udGV4dCwgdXNlU3RhdGUsIFJlYWN0RWxlbWVudCB9IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IHY0IGFzIHV1aWR2NCB9IGZyb20gJ3V1aWQnO1xuXG5pbXBvcnQge1xuICBzaXplcyxcbiAgdmlld1Bvc2l0aW9ucyxcbiAgcGxvdHNQcm9wb3J0aW9uc09wdGlvbnMsXG59IGZyb20gJy4uL2NvbXBvbmVudHMvY29uc3RhbnRzJztcbmltcG9ydCB7XG4gIFNpemVQcm9wcyxcbiAgUGxvdFByb3BzLFxuICBUcmlwbGVQcm9wcyxcbiAgQ3VzdG9taXplUHJvcHMsXG59IGZyb20gJy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcbmltcG9ydCB7IG92ZXJsYXlPcHRpb25zIH0gZnJvbSAnLi4vY29tcG9uZW50cy9jb25zdGFudHMnO1xuXG5leHBvcnQgaW50ZXJmYWNlIExlZnRTaWRlU3RhdGVQcm92aWRlclByb3BzIHtcbiAgY2hpbGRyZW46IFJlYWN0RWxlbWVudDtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBMZWZ0U2lkZVN0YXRlIHtcbiAgc2l6ZTogU2l6ZVByb3BzO1xuICBub3JtYWxpemU6IGJvb2xlYW47XG4gIHN0YXRzOiBib29sZWFuO1xuICBvdmVybGF5UG9zaXRpb246IHN0cmluZztcbiAgb3ZlcmxheTogUGxvdFByb3BzW107XG4gIHRyaXBsZXM6IFRyaXBsZVByb3BzW107XG4gIG92ZXJsYXlQbG90czogVHJpcGxlUHJvcHNbXTtcbiAgd29ya3NwYWNlRm9sZGVyczogc3RyaW5nW107XG4gIG9wZW5PdmVybGF5RGF0YU1lbnU6IGJvb2xlYW47XG4gIHZpZXdQbG90c1Bvc2l0aW9uOiBib29sZWFuO1xuICBsdW1pc2VjdGlvbjogc3RyaW5nIHwgbnVtYmVyO1xuICByaWdodFNpZGVTaXplOiBTaXplUHJvcHM7XG4gIEpTUk9PVG1vZGU6IGJvb2xlYW47XG4gIGN1c3RvbWl6ZVByb3BzOiBDdXN0b21pemVQcm9wcztcbiAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbjogbnVtYmVyO1xufVxuXG5leHBvcnQgY29uc3QgaW5pdGlhbFN0YXRlOiBhbnkgPSB7XG4gIHNpemU6IHNpemVzLm1lZGl1bS5zaXplLFxuICBub3JtYWxpemU6ICdUcnVlJyxcbiAgc3RhdHM6IHRydWUsXG4gIG92ZXJsYXlQb3NpdGlvbjogb3ZlcmxheU9wdGlvbnNbMF0udmFsdWUsXG4gIG92ZXJsYXk6IHVuZGVmaW5lZCxcbiAgb3ZlcmxheVBsb3RzOiBbXSxcbiAgdHJpcGxlczogW10sXG4gIG9wZW5PdmVybGF5RGF0YU1lbnU6IGZhbHNlLFxuICB2aWV3UGxvdHNQb3NpdGlvbjogdmlld1Bvc2l0aW9uc1sxXS52YWx1ZSxcbiAgcHJvcG9ydGlvbjogcGxvdHNQcm9wb3J0aW9uc09wdGlvbnNbMF0udmFsdWUsXG4gIGx1bWlzZWN0aW9uOiAtMSxcbiAgcmlnaHRTaWRlU2l6ZTogc2l6ZXMuZmlsbC5zaXplLFxuICBKU1JPT1Rtb2RlOiBmYWxzZSxcbiAgc2hvcnRjdXRzOiBbXSxcbiAgY3VzdG9taXplUHJvcHM6IHtcbiAgICB4dHlwZTogJycsXG4gICAgeG1pbjogTmFOLFxuICAgIHhtYXg6IE5hTixcbiAgICB5dHlwZTogJycsXG4gICAgeW1pbjogTmFOLFxuICAgIHltYXg6IE5hTixcbiAgICB6dHlwZTogJycsXG4gICAgem1pbjogTmFOLFxuICAgIHptYXg6IE5hTixcbiAgICBkcmF3b3B0czogJycsXG4gICAgd2l0aHJlZjogJycsXG4gIH0sXG4gIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW46IE1hdGgucm91bmQobmV3IERhdGUoKS5nZXRUaW1lKCkgLyAxMDAwMCkgKiAxMCxcbn07XG5cbmV4cG9ydCBpbnRlcmZhY2UgQWN0aW9uUHJvcHMge1xuICB0eXBlOiBzdHJpbmc7XG4gIHBheWxvYWQ6IGFueTtcbn1cblxuY29uc3Qgc3RvcmUgPSBjcmVhdGVDb250ZXh0KGluaXRpYWxTdGF0ZSk7XG5jb25zdCB7IFByb3ZpZGVyIH0gPSBzdG9yZTtcblxuY29uc3QgTGVmdFNpZGVTdGF0ZVByb3ZpZGVyID0gKHsgY2hpbGRyZW4gfTogTGVmdFNpZGVTdGF0ZVByb3ZpZGVyUHJvcHMpID0+IHtcbiAgY29uc3QgW3NpemUsIHNldFNpemVdID0gdXNlU3RhdGU8bnVtYmVyPihpbml0aWFsU3RhdGUuc2l6ZSk7XG4gIGNvbnN0IFtub3JtYWxpemUsIHNldE5vcm1hbGl6ZV0gPSB1c2VTdGF0ZTxib29sZWFuPihpbml0aWFsU3RhdGUubm9ybWFsaXplKTtcbiAgY29uc3QgW3N0YXRzLCBzZXRTdGF0c10gPSB1c2VTdGF0ZTxib29sZWFuPihpbml0aWFsU3RhdGUuc3RhdHMpO1xuICBjb25zdCBbcGxvdHNXaGljaEFyZU92ZXJsYWlkLCBzZXRQbG90c1doaWNoQXJlT3ZlcmxhaWRdID0gdXNlU3RhdGUoe30pO1xuICBjb25zdCBbb3ZlcmxheVBvc2l0aW9uLCBzZXRPdmVybGFpUG9zaXRpb25dID0gdXNlU3RhdGUoXG4gICAgaW5pdGlhbFN0YXRlLm92ZXJsYXlQb3NpdGlvblxuICApO1xuICBjb25zdCBbb3ZlcmxheVBsb3RzLCBzZXRPdmVybGF5XSA9IHVzZVN0YXRlKGluaXRpYWxTdGF0ZS5vdmVybGF5UGxvdHMpO1xuICBjb25zdCBbaW1hZ2VSZWZTY3JvbGxEb3duLCBzZXRJbWFnZVJlZlNjcm9sbERvd25dID0gdXNlU3RhdGUobnVsbCk7XG4gIGNvbnN0IFtwbG90U2VhcmNoRm9sZGVycywgc2V0UGxvdFNlYXJjaEZvbGRlcnNdID0gUmVhY3QudXNlU3RhdGUoW10pO1xuICBjb25zdCBbd29ya3NwYWNlRm9sZGVycywgc2V0V29ya3NwYWNlRm9sZGVyc10gPSBSZWFjdC51c2VTdGF0ZShbXSk7XG4gIGNvbnN0IFt0cmlwbGVzLCBzZXRUcmlwbGVzXSA9IFJlYWN0LnVzZVN0YXRlKGluaXRpYWxTdGF0ZS50cmlwbGVzKTtcbiAgY29uc3QgW29wZW5PdmVybGF5RGF0YU1lbnUsIHRvZ2dsZU92ZXJsYXlEYXRhTWVudV0gPSBSZWFjdC51c2VTdGF0ZShcbiAgICBpbml0aWFsU3RhdGUub3Blbk92ZXJsYXlEYXRhTWVudVxuICApO1xuICBjb25zdCBbdmlld1Bsb3RzUG9zaXRpb24sIHNldFZpZXdQbG90c1Bvc2l0aW9uXSA9IFJlYWN0LnVzZVN0YXRlKFxuICAgIGluaXRpYWxTdGF0ZS52aWV3UGxvdHNQb3NpdGlvblxuICApO1xuICBjb25zdCBbcHJvcG9ydGlvbiwgc2V0UHJvcG9ydGlvbl0gPSBSZWFjdC51c2VTdGF0ZShpbml0aWFsU3RhdGUucHJvcG9ydGlvbik7XG4gIGNvbnN0IFtsdW1pc2VjdGlvbiwgc2V0THVtaXNlY3Rpb25dID0gUmVhY3QudXNlU3RhdGUoXG4gICAgaW5pdGlhbFN0YXRlLmx1bWlzZWN0aW9uXG4gICk7XG5cbiAgY29uc3QgW3JpZ2h0U2lkZVNpemUsIHNldFJpZ2h0U2lkZVNpemVdID0gdXNlU3RhdGU8bnVtYmVyPihcbiAgICBpbml0aWFsU3RhdGUucmlnaHRTaWRlU2l6ZVxuICApO1xuICBjb25zdCBbSlNST09UbW9kZSwgc2V0SlNST09UbW9kZV0gPSB1c2VTdGF0ZTxib29sZWFuPihmYWxzZSk7XG4gIGNvbnN0IFtjdXN0b21pemUsIHNldEN1c3RvbWl6ZV0gPSB1c2VTdGF0ZTxDdXN0b21pemVQcm9wcz4oe1xuICAgIHh0eXBlOiAnJyxcbiAgICB4bWluOiBOYU4sXG4gICAgeG1heDogTmFOLFxuICAgIHl0eXBlOiAnJyxcbiAgICB5bWluOiBOYU4sXG4gICAgeW1heDogTmFOLFxuICAgIHp0eXBlOiAnJyxcbiAgICB6bWluOiBOYU4sXG4gICAgem1heDogTmFOLFxuICAgIGRyYXdvcHRzOiAnJyxcbiAgICB3aXRocmVmOiAnJyxcbiAgfSk7XG5cbiAgY29uc3QgW3J1bnNfc2V0X2Zvcl9vdmVybGF5LCBzZXRfcnVuc19zZXRfZm9yX292ZXJsYXldID0gUmVhY3QudXNlU3RhdGU8XG4gICAgVHJpcGxlUHJvcHNbXVxuICA+KHRyaXBsZXMgPyB0cmlwbGVzIDogW10pO1xuICBjb25zdCBbdXBkYXRlLCBzZXRfdXBkYXRlXSA9IHVzZVN0YXRlPGJvb2xlYW4+KGZhbHNlKTtcblxuICBjb25zdCBjaGFuZ2VfdmFsdWVfaW5fcmVmZXJlbmNlX3RhYmxlID0gKFxuICAgIHZhbHVlOiBzdHJpbmcgfCBudW1iZXIsXG4gICAga2V5OiBzdHJpbmcsXG4gICAgaWQ6IHN0cmluZyB8IG51bWJlciB8IGJvb2xlYW5cbiAgKSA9PiB7XG4gICAgY29uc3QgY29weSA9IFsuLi50cmlwbGVzXTtcbiAgICAvL3RyaXBsZXMgYXJlIHRob3NlIHJ1bnMgd2hpY2ggYXJlIGFscmVhZHkgb3ZlcmxhaWQuXG4gICAgLy9ydW5zX3NldF9mb3Jfb3ZlcmxheSBhcmUgcnVucyB3aGljaCBhcmUgc2VrZWN0ZWQgZm9yIG92ZXJsYXksXG4gICAgLy9idXQgbm90IG92ZXJsYWlkIHlldC5cbiAgICBsZXQgY3VycmVudF9saW5lOiBUcmlwbGVQcm9wcyA9IHRyaXBsZXMuZmlsdGVyKFxuICAgICAgKGxpbmU6IFRyaXBsZVByb3BzKSA9PiBsaW5lLmlkID09PSBpZFxuICAgIClbMF07XG4gICAgaWYgKCFjdXJyZW50X2xpbmUpIHtcbiAgICAgIGN1cnJlbnRfbGluZSA9IHJ1bnNfc2V0X2Zvcl9vdmVybGF5LmZpbHRlcihcbiAgICAgICAgKGxpbmU6IFRyaXBsZVByb3BzKSA9PiBsaW5lLmlkID09PSBpZFxuICAgICAgKVswXTtcbiAgICB9XG5cbiAgICBjb25zdCBpbmRleF9vZl9saW5lOiBudW1iZXIgPSBjb3B5LmluZGV4T2YoY3VycmVudF9saW5lKTtcbiAgICBjdXJyZW50X2xpbmVba2V5XSA9IHZhbHVlO1xuICAgIGNvcHlbaW5kZXhfb2ZfbGluZV0gPSBjdXJyZW50X2xpbmU7XG4gICAgc2V0VHJpcGxlcyhjb3B5KTtcbiAgfTtcblxuICBjb25zdCBbdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiwgc2V0X3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW5dID0gdXNlU3RhdGUoXG4gICAgaW5pdGlhbFN0YXRlLnVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW5cbiAgKTtcblxuICByZXR1cm4gKFxuICAgIDxQcm92aWRlclxuICAgICAgdmFsdWU9e3tcbiAgICAgICAgc2l6ZSxcbiAgICAgICAgc2V0U2l6ZSxcbiAgICAgICAgbm9ybWFsaXplLFxuICAgICAgICBzZXROb3JtYWxpemUsXG4gICAgICAgIHN0YXRzLFxuICAgICAgICBzZXRTdGF0cyxcbiAgICAgICAgcGxvdHNXaGljaEFyZU92ZXJsYWlkLFxuICAgICAgICBzZXRQbG90c1doaWNoQXJlT3ZlcmxhaWQsXG4gICAgICAgIG92ZXJsYXlQb3NpdGlvbixcbiAgICAgICAgc2V0T3ZlcmxhaVBvc2l0aW9uLFxuICAgICAgICBvdmVybGF5UGxvdHMsXG4gICAgICAgIHNldE92ZXJsYXksXG4gICAgICAgIGltYWdlUmVmU2Nyb2xsRG93bixcbiAgICAgICAgc2V0SW1hZ2VSZWZTY3JvbGxEb3duLFxuICAgICAgICB3b3Jrc3BhY2VGb2xkZXJzLFxuICAgICAgICBzZXRXb3Jrc3BhY2VGb2xkZXJzLFxuICAgICAgICBwbG90U2VhcmNoRm9sZGVycyxcbiAgICAgICAgc2V0UGxvdFNlYXJjaEZvbGRlcnMsXG4gICAgICAgIGNoYW5nZV92YWx1ZV9pbl9yZWZlcmVuY2VfdGFibGUsXG4gICAgICAgIHRyaXBsZXMsXG4gICAgICAgIHNldFRyaXBsZXMsXG4gICAgICAgIG9wZW5PdmVybGF5RGF0YU1lbnUsXG4gICAgICAgIHRvZ2dsZU92ZXJsYXlEYXRhTWVudSxcbiAgICAgICAgdmlld1Bsb3RzUG9zaXRpb24sXG4gICAgICAgIHNldFZpZXdQbG90c1Bvc2l0aW9uLFxuICAgICAgICBwcm9wb3J0aW9uLFxuICAgICAgICBzZXRQcm9wb3J0aW9uLFxuICAgICAgICBsdW1pc2VjdGlvbixcbiAgICAgICAgc2V0THVtaXNlY3Rpb24sXG4gICAgICAgIHJpZ2h0U2lkZVNpemUsXG4gICAgICAgIHNldFJpZ2h0U2lkZVNpemUsXG4gICAgICAgIEpTUk9PVG1vZGUsXG4gICAgICAgIHNldEpTUk9PVG1vZGUsXG4gICAgICAgIGN1c3RvbWl6ZSxcbiAgICAgICAgc2V0Q3VzdG9taXplLFxuICAgICAgICBydW5zX3NldF9mb3Jfb3ZlcmxheSxcbiAgICAgICAgc2V0X3J1bnNfc2V0X2Zvcl9vdmVybGF5LFxuICAgICAgICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuLFxuICAgICAgICBzZXRfdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbixcbiAgICAgICAgdXBkYXRlLFxuICAgICAgICBzZXRfdXBkYXRlLFxuICAgICAgfX1cbiAgICA+XG4gICAgICB7Y2hpbGRyZW59XG4gICAgPC9Qcm92aWRlcj5cbiAgKTtcbn07XG5cbmV4cG9ydCB7IHN0b3JlLCBMZWZ0U2lkZVN0YXRlUHJvdmlkZXIgfTtcbiJdLCJzb3VyY2VSb290IjoiIn0=