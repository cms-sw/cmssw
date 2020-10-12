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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0LnRzeCJdLCJuYW1lcyI6WyJpbml0aWFsU3RhdGUiLCJzaXplIiwic2l6ZXMiLCJtZWRpdW0iLCJub3JtYWxpemUiLCJzdGF0cyIsIm92ZXJsYXlQb3NpdGlvbiIsIm92ZXJsYXlPcHRpb25zIiwidmFsdWUiLCJvdmVybGF5IiwidW5kZWZpbmVkIiwib3ZlcmxheVBsb3RzIiwidHJpcGxlcyIsIm9wZW5PdmVybGF5RGF0YU1lbnUiLCJ2aWV3UGxvdHNQb3NpdGlvbiIsInZpZXdQb3NpdGlvbnMiLCJwcm9wb3J0aW9uIiwicGxvdHNQcm9wb3J0aW9uc09wdGlvbnMiLCJsdW1pc2VjdGlvbiIsInJpZ2h0U2lkZVNpemUiLCJmaWxsIiwiSlNST09UbW9kZSIsInNob3J0Y3V0cyIsImN1c3RvbWl6ZVByb3BzIiwieHR5cGUiLCJ4bWluIiwiTmFOIiwieG1heCIsInl0eXBlIiwieW1pbiIsInltYXgiLCJ6dHlwZSIsInptaW4iLCJ6bWF4IiwiZHJhd29wdHMiLCJ3aXRocmVmIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsIk1hdGgiLCJyb3VuZCIsIkRhdGUiLCJnZXRUaW1lIiwic3RvcmUiLCJjcmVhdGVDb250ZXh0IiwiUHJvdmlkZXIiLCJMZWZ0U2lkZVN0YXRlUHJvdmlkZXIiLCJjaGlsZHJlbiIsInVzZVN0YXRlIiwic2V0U2l6ZSIsInNldE5vcm1hbGl6ZSIsInNldFN0YXRzIiwicGxvdHNXaGljaEFyZU92ZXJsYWlkIiwic2V0UGxvdHNXaGljaEFyZU92ZXJsYWlkIiwic2V0T3ZlcmxhaVBvc2l0aW9uIiwic2V0T3ZlcmxheSIsImltYWdlUmVmU2Nyb2xsRG93biIsInNldEltYWdlUmVmU2Nyb2xsRG93biIsIlJlYWN0IiwicGxvdFNlYXJjaEZvbGRlcnMiLCJzZXRQbG90U2VhcmNoRm9sZGVycyIsIndvcmtzcGFjZUZvbGRlcnMiLCJzZXRXb3Jrc3BhY2VGb2xkZXJzIiwic2V0VHJpcGxlcyIsInRvZ2dsZU92ZXJsYXlEYXRhTWVudSIsInNldFZpZXdQbG90c1Bvc2l0aW9uIiwic2V0UHJvcG9ydGlvbiIsInNldEx1bWlzZWN0aW9uIiwic2V0UmlnaHRTaWRlU2l6ZSIsInNldEpTUk9PVG1vZGUiLCJjdXN0b21pemUiLCJzZXRDdXN0b21pemUiLCJydW5zX3NldF9mb3Jfb3ZlcmxheSIsInNldF9ydW5zX3NldF9mb3Jfb3ZlcmxheSIsInVwZGF0ZSIsInNldF91cGRhdGUiLCJjaGFuZ2VfdmFsdWVfaW5fcmVmZXJlbmNlX3RhYmxlIiwia2V5IiwiaWQiLCJjb3B5IiwiY3VycmVudF9saW5lIiwiZmlsdGVyIiwibGluZSIsImluZGV4X29mX2xpbmUiLCJpbmRleE9mIiwic2V0X3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4iXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBR0E7QUFXQTtBQXdCTyxJQUFNQSxZQUFpQixHQUFHO0FBQy9CQyxNQUFJLEVBQUVDLDJEQUFLLENBQUNDLE1BQU4sQ0FBYUYsSUFEWTtBQUUvQkcsV0FBUyxFQUFFLE1BRm9CO0FBRy9CQyxPQUFLLEVBQUUsSUFId0I7QUFJL0JDLGlCQUFlLEVBQUVDLG9FQUFjLENBQUMsQ0FBRCxDQUFkLENBQWtCQyxLQUpKO0FBSy9CQyxTQUFPLEVBQUVDLFNBTHNCO0FBTS9CQyxjQUFZLEVBQUUsRUFOaUI7QUFPL0JDLFNBQU8sRUFBRSxFQVBzQjtBQVEvQkMscUJBQW1CLEVBQUUsS0FSVTtBQVMvQkMsbUJBQWlCLEVBQUVDLG1FQUFhLENBQUMsQ0FBRCxDQUFiLENBQWlCUCxLQVRMO0FBVS9CUSxZQUFVLEVBQUVDLDZFQUF1QixDQUFDLENBQUQsQ0FBdkIsQ0FBMkJULEtBVlI7QUFXL0JVLGFBQVcsRUFBRSxDQUFDLENBWGlCO0FBWS9CQyxlQUFhLEVBQUVqQiwyREFBSyxDQUFDa0IsSUFBTixDQUFXbkIsSUFaSztBQWEvQm9CLFlBQVUsRUFBRSxLQWJtQjtBQWMvQkMsV0FB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ZTg0OWVkYjEzNGE2NjYxOGYwZS5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0LCB7IGNyZWF0ZUNvbnRleHQsIHVzZVN0YXRlLCBSZWFjdEVsZW1lbnQgfSBmcm9tICdyZWFjdCc7XG5pbXBvcnQgeyB2NCBhcyB1dWlkdjQgfSBmcm9tICd1dWlkJztcblxuaW1wb3J0IHtcbiAgc2l6ZXMsXG4gIHZpZXdQb3NpdGlvbnMsXG4gIHBsb3RzUHJvcG9ydGlvbnNPcHRpb25zLFxufSBmcm9tICcuLi9jb21wb25lbnRzL2NvbnN0YW50cyc7XG5pbXBvcnQge1xuICBTaXplUHJvcHMsXG4gIFBsb3RQcm9wcyxcbiAgVHJpcGxlUHJvcHMsXG4gIEN1c3RvbWl6ZVByb3BzLFxufSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XG5pbXBvcnQgeyBvdmVybGF5T3B0aW9ucyB9IGZyb20gJy4uL2NvbXBvbmVudHMvY29uc3RhbnRzJztcblxuZXhwb3J0IGludGVyZmFjZSBMZWZ0U2lkZVN0YXRlUHJvdmlkZXJQcm9wcyB7XG4gIGNoaWxkcmVuOiBSZWFjdEVsZW1lbnQ7XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgTGVmdFNpZGVTdGF0ZSB7XG4gIHNpemU6IFNpemVQcm9wcztcbiAgbm9ybWFsaXplOiBib29sZWFuO1xuICBzdGF0czogYm9vbGVhbjtcbiAgb3ZlcmxheVBvc2l0aW9uOiBzdHJpbmc7XG4gIG92ZXJsYXk6IFBsb3RQcm9wc1tdO1xuICB0cmlwbGVzOiBUcmlwbGVQcm9wc1tdO1xuICBvdmVybGF5UGxvdHM6IFRyaXBsZVByb3BzW107XG4gIHdvcmtzcGFjZUZvbGRlcnM6IHN0cmluZ1tdO1xuICBvcGVuT3ZlcmxheURhdGFNZW51OiBib29sZWFuO1xuICB2aWV3UGxvdHNQb3NpdGlvbjogYm9vbGVhbjtcbiAgbHVtaXNlY3Rpb246IHN0cmluZyB8IG51bWJlcjtcbiAgcmlnaHRTaWRlU2l6ZTogU2l6ZVByb3BzO1xuICBKU1JPT1Rtb2RlOiBib29sZWFuO1xuICBjdXN0b21pemVQcm9wczogQ3VzdG9taXplUHJvcHM7XG4gIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW46IG51bWJlcjtcbn1cblxuZXhwb3J0IGNvbnN0IGluaXRpYWxTdGF0ZTogYW55ID0ge1xuICBzaXplOiBzaXplcy5tZWRpdW0uc2l6ZSxcbiAgbm9ybWFsaXplOiAnVHJ1ZScsXG4gIHN0YXRzOiB0cnVlLFxuICBvdmVybGF5UG9zaXRpb246IG92ZXJsYXlPcHRpb25zWzBdLnZhbHVlLFxuICBvdmVybGF5OiB1bmRlZmluZWQsXG4gIG92ZXJsYXlQbG90czogW10sXG4gIHRyaXBsZXM6IFtdLFxuICBvcGVuT3ZlcmxheURhdGFNZW51OiBmYWxzZSxcbiAgdmlld1Bsb3RzUG9zaXRpb246IHZpZXdQb3NpdGlvbnNbMV0udmFsdWUsXG4gIHByb3BvcnRpb246IHBsb3RzUHJvcG9ydGlvbnNPcHRpb25zWzBdLnZhbHVlLFxuICBsdW1pc2VjdGlvbjogLTEsXG4gIHJpZ2h0U2lkZVNpemU6IHNpemVzLmZpbGwuc2l6ZSxcbiAgSlNST09UbW9kZTogZmFsc2UsXG4gIHNob3J0Y3V0czogW10sXG4gIGN1c3RvbWl6ZVByb3BzOiB7XG4gICAgeHR5cGU6ICcnLFxuICAgIHhtaW46IE5hTixcbiAgICB4bWF4OiBOYU4sXG4gICAgeXR5cGU6ICcnLFxuICAgIHltaW46IE5hTixcbiAgICB5bWF4OiBOYU4sXG4gICAgenR5cGU6ICcnLFxuICAgIHptaW46IE5hTixcbiAgICB6bWF4OiBOYU4sXG4gICAgZHJhd29wdHM6ICcnLFxuICAgIHdpdGhyZWY6ICcnLFxuICB9LFxuICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuOiBNYXRoLnJvdW5kKG5ldyBEYXRlKCkuZ2V0VGltZSgpIC8gMTAwMDApICogMTAsXG59O1xuXG5leHBvcnQgaW50ZXJmYWNlIEFjdGlvblByb3BzIHtcbiAgdHlwZTogc3RyaW5nO1xuICBwYXlsb2FkOiBhbnk7XG59XG5cbmNvbnN0IHN0b3JlID0gY3JlYXRlQ29udGV4dChpbml0aWFsU3RhdGUpO1xuY29uc3QgeyBQcm92aWRlciB9ID0gc3RvcmU7XG5cbmNvbnN0IExlZnRTaWRlU3RhdGVQcm92aWRlciA9ICh7IGNoaWxkcmVuIH06IExlZnRTaWRlU3RhdGVQcm92aWRlclByb3BzKSA9PiB7XG4gIGNvbnN0IFtzaXplLCBzZXRTaXplXSA9IHVzZVN0YXRlPG51bWJlcj4oaW5pdGlhbFN0YXRlLnNpemUpO1xuICBjb25zdCBbbm9ybWFsaXplLCBzZXROb3JtYWxpemVdID0gdXNlU3RhdGU8Ym9vbGVhbj4oaW5pdGlhbFN0YXRlLm5vcm1hbGl6ZSk7XG4gIGNvbnN0IFtzdGF0cywgc2V0U3RhdHNdID0gdXNlU3RhdGU8Ym9vbGVhbj4oaW5pdGlhbFN0YXRlLnN0YXRzKTtcbiAgY29uc3QgW3Bsb3RzV2hpY2hBcmVPdmVybGFpZCwgc2V0UGxvdHNXaGljaEFyZU92ZXJsYWlkXSA9IHVzZVN0YXRlKHt9KTtcbiAgY29uc3QgW292ZXJsYXlQb3NpdGlvbiwgc2V0T3ZlcmxhaVBvc2l0aW9uXSA9IHVzZVN0YXRlKFxuICAgIGluaXRpYWxTdGF0ZS5vdmVybGF5UG9zaXRpb25cbiAgKTtcbiAgY29uc3QgW292ZXJsYXlQbG90cywgc2V0T3ZlcmxheV0gPSB1c2VTdGF0ZShpbml0aWFsU3RhdGUub3ZlcmxheVBsb3RzKTtcbiAgY29uc3QgW2ltYWdlUmVmU2Nyb2xsRG93biwgc2V0SW1hZ2VSZWZTY3JvbGxEb3duXSA9IHVzZVN0YXRlKG51bGwpO1xuICBjb25zdCBbcGxvdFNlYXJjaEZvbGRlcnMsIHNldFBsb3RTZWFyY2hGb2xkZXJzXSA9IFJlYWN0LnVzZVN0YXRlKFtdKTtcbiAgY29uc3QgW3dvcmtzcGFjZUZvbGRlcnMsIHNldFdvcmtzcGFjZUZvbGRlcnNdID0gUmVhY3QudXNlU3RhdGUoW10pO1xuICBjb25zdCBbdHJpcGxlcywgc2V0VHJpcGxlc10gPSBSZWFjdC51c2VTdGF0ZShpbml0aWFsU3RhdGUudHJpcGxlcyk7XG4gIGNvbnN0IFtvcGVuT3ZlcmxheURhdGFNZW51LCB0b2dnbGVPdmVybGF5RGF0YU1lbnVdID0gUmVhY3QudXNlU3RhdGUoXG4gICAgaW5pdGlhbFN0YXRlLm9wZW5PdmVybGF5RGF0YU1lbnVcbiAgKTtcbiAgY29uc3QgW3ZpZXdQbG90c1Bvc2l0aW9uLCBzZXRWaWV3UGxvdHNQb3NpdGlvbl0gPSBSZWFjdC51c2VTdGF0ZShcbiAgICBpbml0aWFsU3RhdGUudmlld1Bsb3RzUG9zaXRpb25cbiAgKTtcbiAgY29uc3QgW3Byb3BvcnRpb24sIHNldFByb3BvcnRpb25dID0gUmVhY3QudXNlU3RhdGUoaW5pdGlhbFN0YXRlLnByb3BvcnRpb24pO1xuICBjb25zdCBbbHVtaXNlY3Rpb24sIHNldEx1bWlzZWN0aW9uXSA9IFJlYWN0LnVzZVN0YXRlKFxuICAgIGluaXRpYWxTdGF0ZS5sdW1pc2VjdGlvblxuICApO1xuXG4gIGNvbnN0IFtyaWdodFNpZGVTaXplLCBzZXRSaWdodFNpZGVTaXplXSA9IHVzZVN0YXRlPG51bWJlcj4oXG4gICAgaW5pdGlhbFN0YXRlLnJpZ2h0U2lkZVNpemVcbiAgKTtcbiAgY29uc3QgW0pTUk9PVG1vZGUsIHNldEpTUk9PVG1vZGVdID0gdXNlU3RhdGU8Ym9vbGVhbj4oZmFsc2UpO1xuICBjb25zdCBbY3VzdG9taXplLCBzZXRDdXN0b21pemVdID0gdXNlU3RhdGU8Q3VzdG9taXplUHJvcHM+KHtcbiAgICB4dHlwZTogJycsXG4gICAgeG1pbjogTmFOLFxuICAgIHhtYXg6IE5hTixcbiAgICB5dHlwZTogJycsXG4gICAgeW1pbjogTmFOLFxuICAgIHltYXg6IE5hTixcbiAgICB6dHlwZTogJycsXG4gICAgem1pbjogTmFOLFxuICAgIHptYXg6IE5hTixcbiAgICBkcmF3b3B0czogJycsXG4gICAgd2l0aHJlZjogJycsXG4gIH0pO1xuXG4gIGNvbnN0IFtydW5zX3NldF9mb3Jfb3ZlcmxheSwgc2V0X3J1bnNfc2V0X2Zvcl9vdmVybGF5XSA9IFJlYWN0LnVzZVN0YXRlPFxuICAgIFRyaXBsZVByb3BzW11cbiAgPih0cmlwbGVzID8gdHJpcGxlcyA6IFtdKTtcbiAgY29uc3QgW3VwZGF0ZSwgc2V0X3VwZGF0ZV0gPSB1c2VTdGF0ZTxib29sZWFuPihmYWxzZSk7XG5cbiAgY29uc3QgY2hhbmdlX3ZhbHVlX2luX3JlZmVyZW5jZV90YWJsZSA9IChcbiAgICB2YWx1ZTogc3RyaW5nIHwgbnVtYmVyLFxuICAgIGtleTogc3RyaW5nLFxuICAgIGlkOiBzdHJpbmcgfCBudW1iZXIgfCBib29sZWFuXG4gICkgPT4ge1xuICAgIGNvbnN0IGNvcHkgPSBbLi4udHJpcGxlc107XG4gICAgLy90cmlwbGVzIGFyZSB0aG9zZSBydW5zIHdoaWNoIGFyZSBhbHJlYWR5IG92ZXJsYWlkLlxuICAgIC8vcnVuc19zZXRfZm9yX292ZXJsYXkgYXJlIHJ1bnMgd2hpY2ggYXJlIHNla2VjdGVkIGZvciBvdmVybGF5LFxuICAgIC8vYnV0IG5vdCBvdmVybGFpZCB5ZXQuXG4gICAgbGV0IGN1cnJlbnRfbGluZTogVHJpcGxlUHJvcHMgPSB0cmlwbGVzLmZpbHRlcihcbiAgICAgIChsaW5lOiBUcmlwbGVQcm9wcykgPT4gbGluZS5pZCA9PT0gaWRcbiAgICApWzBdO1xuICAgIGlmICghY3VycmVudF9saW5lKSB7XG4gICAgICBjdXJyZW50X2xpbmUgPSBydW5zX3NldF9mb3Jfb3ZlcmxheS5maWx0ZXIoXG4gICAgICAgIChsaW5lOiBUcmlwbGVQcm9wcykgPT4gbGluZS5pZCA9PT0gaWRcbiAgICAgIClbMF07XG4gICAgfVxuXG4gICAgY29uc3QgaW5kZXhfb2ZfbGluZTogbnVtYmVyID0gY29weS5pbmRleE9mKGN1cnJlbnRfbGluZSk7XG4gICAgY3VycmVudF9saW5lW2tleV0gPSB2YWx1ZTtcbiAgICBjb3B5W2luZGV4X29mX2xpbmVdID0gY3VycmVudF9saW5lO1xuICAgIHNldFRyaXBsZXMoY29weSk7XG4gIH07XG5cbiAgY29uc3QgW3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4sIHNldF91cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuXSA9IHVzZVN0YXRlKFxuICAgIGluaXRpYWxTdGF0ZS51cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuXG4gICk7XG5cbiAgcmV0dXJuIChcbiAgICA8UHJvdmlkZXJcbiAgICAgIHZhbHVlPXt7XG4gICAgICAgIHNpemUsXG4gICAgICAgIHNldFNpemUsXG4gICAgICAgIG5vcm1hbGl6ZSxcbiAgICAgICAgc2V0Tm9ybWFsaXplLFxuICAgICAgICBzdGF0cyxcbiAgICAgICAgc2V0U3RhdHMsXG4gICAgICAgIHBsb3RzV2hpY2hBcmVPdmVybGFpZCxcbiAgICAgICAgc2V0UGxvdHNXaGljaEFyZU92ZXJsYWlkLFxuICAgICAgICBvdmVybGF5UG9zaXRpb24sXG4gICAgICAgIHNldE92ZXJsYWlQb3NpdGlvbixcbiAgICAgICAgb3ZlcmxheVBsb3RzLFxuICAgICAgICBzZXRPdmVybGF5LFxuICAgICAgICBpbWFnZVJlZlNjcm9sbERvd24sXG4gICAgICAgIHNldEltYWdlUmVmU2Nyb2xsRG93bixcbiAgICAgICAgd29ya3NwYWNlRm9sZGVycyxcbiAgICAgICAgc2V0V29ya3NwYWNlRm9sZGVycyxcbiAgICAgICAgcGxvdFNlYXJjaEZvbGRlcnMsXG4gICAgICAgIHNldFBsb3RTZWFyY2hGb2xkZXJzLFxuICAgICAgICBjaGFuZ2VfdmFsdWVfaW5fcmVmZXJlbmNlX3RhYmxlLFxuICAgICAgICB0cmlwbGVzLFxuICAgICAgICBzZXRUcmlwbGVzLFxuICAgICAgICBvcGVuT3ZlcmxheURhdGFNZW51LFxuICAgICAgICB0b2dnbGVPdmVybGF5RGF0YU1lbnUsXG4gICAgICAgIHZpZXdQbG90c1Bvc2l0aW9uLFxuICAgICAgICBzZXRWaWV3UGxvdHNQb3NpdGlvbixcbiAgICAgICAgcHJvcG9ydGlvbixcbiAgICAgICAgc2V0UHJvcG9ydGlvbixcbiAgICAgICAgbHVtaXNlY3Rpb24sXG4gICAgICAgIHNldEx1bWlzZWN0aW9uLFxuICAgICAgICByaWdodFNpZGVTaXplLFxuICAgICAgICBzZXRSaWdodFNpZGVTaXplLFxuICAgICAgICBKU1JPT1Rtb2RlLFxuICAgICAgICBzZXRKU1JPT1Rtb2RlLFxuICAgICAgICBjdXN0b21pemUsXG4gICAgICAgIHNldEN1c3RvbWl6ZSxcbiAgICAgICAgcnVuc19zZXRfZm9yX292ZXJsYXksXG4gICAgICAgIHNldF9ydW5zX3NldF9mb3Jfb3ZlcmxheSxcbiAgICAgICAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbixcbiAgICAgICAgc2V0X3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4sXG4gICAgICAgIHVwZGF0ZSxcbiAgICAgICAgc2V0X3VwZGF0ZSxcbiAgICAgIH19XG4gICAgPlxuICAgICAge2NoaWxkcmVufVxuICAgIDwvUHJvdmlkZXI+XG4gICk7XG59O1xuXG5leHBvcnQgeyBzdG9yZSwgTGVmdFNpZGVTdGF0ZVByb3ZpZGVyIH07XG4iXSwic291cmNlUm9vdCI6IiJ9