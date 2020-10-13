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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0LnRzeCJdLCJuYW1lcyI6WyJpbml0aWFsU3RhdGUiLCJzaXplIiwic2l6ZXMiLCJtZWRpdW0iLCJub3JtYWxpemUiLCJzdGF0cyIsIm92ZXJsYXlQb3NpdGlvbiIsIm92ZXJsYXlPcHRpb25zIiwidmFsdWUiLCJvdmVybGF5IiwidW5kZWZpbmVkIiwib3ZlcmxheVBsb3RzIiwidHJpcGxlcyIsIm9wZW5PdmVybGF5RGF0YU1lbnUiLCJ2aWV3UGxvdHNQb3NpdGlvbiIsInZpZXdQb3NpdGlvbnMiLCJwcm9wb3J0aW9uIiwicGxvdHNQcm9wb3J0aW9uc09wdGlvbnMiLCJsdW1pc2VjdGlvbiIsInJpZ2h0U2lkZVNpemUiLCJmaWxsIiwiSlNST09UbW9kZSIsInNob3J0Y3V0cyIsImN1c3RvbWl6ZVByb3BzIiwieHR5cGUiLCJ4bWluIiwiTmFOIiwieG1heCIsInl0eXBlIiwieW1pbiIsInltYXgiLCJ6dHlwZSIsInptaW4iLCJ6bWF4IiwiZHJhd29wdHMiLCJ3aXRocmVmIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsIk1hdGgiLCJyb3VuZCIsIkRhdGUiLCJnZXRUaW1lIiwic3RvcmUiLCJjcmVhdGVDb250ZXh0IiwiUHJvdmlkZXIiLCJMZWZ0U2lkZVN0YXRlUHJvdmlkZXIiLCJjaGlsZHJlbiIsInVzZVN0YXRlIiwic2V0U2l6ZSIsInNldE5vcm1hbGl6ZSIsInNldFN0YXRzIiwicGxvdHNXaGljaEFyZU92ZXJsYWlkIiwic2V0UGxvdHNXaGljaEFyZU92ZXJsYWlkIiwic2V0T3ZlcmxhaVBvc2l0aW9uIiwic2V0T3ZlcmxheSIsImltYWdlUmVmU2Nyb2xsRG93biIsInNldEltYWdlUmVmU2Nyb2xsRG93biIsIlJlYWN0IiwicGxvdFNlYXJjaEZvbGRlcnMiLCJzZXRQbG90U2VhcmNoRm9sZGVycyIsIndvcmtzcGFjZXMiLCJsYWJlbCIsIndvcmtzcGFjZSIsInNldFdvcmtzcGFjZSIsInNldFRyaXBsZXMiLCJ0b2dnbGVPdmVybGF5RGF0YU1lbnUiLCJzZXRWaWV3UGxvdHNQb3NpdGlvbiIsInNldFByb3BvcnRpb24iLCJzZXRMdW1pc2VjdGlvbiIsInNldFJpZ2h0U2lkZVNpemUiLCJzZXRKU1JPT1Rtb2RlIiwiY3VzdG9taXplIiwic2V0Q3VzdG9taXplIiwicnVuc19zZXRfZm9yX292ZXJsYXkiLCJzZXRfcnVuc19zZXRfZm9yX292ZXJsYXkiLCJ1cGRhdGUiLCJzZXRfdXBkYXRlIiwiY2hhbmdlX3ZhbHVlX2luX3JlZmVyZW5jZV90YWJsZSIsImtleSIsImlkIiwiY29weSIsImN1cnJlbnRfbGluZSIsImZpbHRlciIsImxpbmUiLCJpbmRleF9vZl9saW5lIiwiaW5kZXhPZiIsInNldF91cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFHQTtBQVdBO0FBQ0E7QUF3Qk8sSUFBTUEsWUFBaUIsR0FBRztBQUMvQkMsTUFBSSxFQUFFQywyREFBSyxDQUFDQyxNQUFOLENBQWFGLElBRFk7QUFFL0JHLFdBQVMsRUFBRSxNQUZvQjtBQUcvQkMsT0FBSyxFQUFFLElBSHdCO0FBSS9CQyxpQkFBZSxFQUFFQyxvRUFBYyxDQUFDLENBQUQsQ0FBZCxDQUFrQkMsS0FKSjtBQUsvQkMsU0FBTyxFQUFFQyxTQUxzQjtBQU0vQkMsY0FBWSxFQUFFLEVBTmlCO0FBTy9CQyxTQUFPLEVBQUUsRUFQc0I7QUFRL0JDLHFCQUFtQixFQUFFLEtBUlU7QUFTL0JDLG1CQUFpQixFQUFFQyxtRUFBYSxDQUFDLENBQUQsQ0FBYixDQUFpQlAsS0FUTDtBQVUvQlEsWUFBVSxFQUFFQyw2RUFBdUIsQ0FBQyxDQUFELENBQXZCLENBQTJCVCxLQVZSO0FBVy9CVSxhQUFXLEVBQUUsQ0FBQyxDQVhpQjtBQVkvQkMsZUFBYSxFQUFFakIsMkRBQUssQ0FBQ2tCLElBQU4sQ0FBV25CLElBWks7QUFhL0JvQixZQUFVLEVBQUUsS0FibUI7QUFjL0JDLFdBQVMsRUFBRSxFQWRvQjtBQWUvQkMsZ0JBQWMsRUFBRTtBQUNkQyxTQUFLLEVBQUUsRUFETztBQUVkQyxRQUFJLEVBQUVDLEdBRlE7QUFHZEMsUUFBSSxFQUFFRCxHQUhRO0FBSWRFLFNBQUssRUFBRSxFQUpPO0FBS2RDLFFBQUksRUFBRUgsR0FMUTtBQU1kSSxRQUFJLEVBQUVKLEdBTlE7QUFPZEssU0FBSyxFQUFFLEVBUE87QUFRZEMsUUFBSSxFQUFFTixHQVJRO0FBU2RPLFFBQUksRUFBRVAsR0FUUTtBQVVkUSxZQUFRLEVBQUUsRUFWSTtBQVdkQyxXQUFPLEVBQUU7QUFYSyxHQWZlO0FBNEIvQkMsMkJBQXlCLEVBQUVDLElBQUksQ0FBQ0MsS0FBTCxDQUFXLElBQUlDLElBQUosR0FBV0MsT0FBWCxLQUF1QixLQUFsQyxJQUEyQztBQTVCdkMsQ0FBMUI7QUFvQ1AsSUFBTUMsS0FBSyxnQkFBR0MsMkRBQWEsQ0FBQzFDLFlBQUQsQ0FBM0I7SUFDUTJDLFEsR0FBYUYsSyxDQUFiRSxROztBQUVSLElBQU1DLHFCQUFxQixHQUFHLFNBQXhCQSxxQkFBd0IsT0FBOEM7QUFBQTs7QUFBQSxNQUEzQ0MsUUFBMkMsUUFBM0NBLFFBQTJDOztBQUFBLGtCQUNsREMsc0RBQVEsQ0FBUzlDLFlBQVksQ0FBQ0MsSUFBdEIsQ0FEMEM7QUFBQSxNQUNuRUEsSUFEbUU7QUFBQSxNQUM3RDhDLE9BRDZEOztBQUFBLG1CQUV4Q0Qsc0RBQVEsQ0FBVTlDLFlBQVksQ0FBQ0ksU0FBdkIsQ0FGZ0M7QUFBQSxNQUVuRUEsU0FGbUU7QUFBQSxNQUV4RDRDLFlBRndEOztBQUFBLG1CQUdoREYsc0RBQVEsQ0FBVTlDLFlBQVksQ0FBQ0ssS0FBdkIsQ0FId0M7QUFBQSxNQUduRUEsS0FIbUU7QUFBQSxNQUc1RDRDLFFBSDREOztBQUFBLG1CQUloQkgsc0RBQVEsQ0FBQyxFQUFELENBSlE7QUFBQSxNQUluRUkscUJBSm1FO0FBQUEsTUFJNUNDLHdCQUo0Qzs7QUFBQSxtQkFLNUJMLHNEQUFRLENBQ3BEOUMsWUFBWSxDQUFDTSxlQUR1QyxDQUxvQjtBQUFBLE1BS25FQSxlQUxtRTtBQUFBLE1BS2xEOEMsa0JBTGtEOztBQUFBLG1CQVF2Q04sc0RBQVEsQ0FBQzlDLFlBQVksQ0FBQ1csWUFBZCxDQVIrQjtBQUFBLE1BUW5FQSxZQVJtRTtBQUFBLE1BUXJEMEMsVUFScUQ7O0FBQUEsbUJBU3RCUCxzREFBUSxDQUFDLElBQUQsQ0FUYztBQUFBLE1BU25FUSxrQkFUbUU7QUFBQSxNQVMvQ0MscUJBVCtDOztBQUFBLHdCQVV4QkMsNENBQUssQ0FBQ1YsUUFBTixDQUFlLEVBQWYsQ0FWd0I7QUFBQTtBQUFBLE1BVW5FVyxpQkFWbUU7QUFBQSxNQVVoREMsb0JBVmdEOztBQUFBLHlCQVd4Q0YsNENBQUssQ0FBQ1YsUUFBTixDQUFlYSw4REFBVSxDQUFDLENBQUQsQ0FBVixDQUFjQSxVQUFkLENBQXlCLENBQXpCLEVBQTRCQyxLQUEzQyxDQVh3QztBQUFBO0FBQUEsTUFXbkVDLFNBWG1FO0FBQUEsTUFXeERDLFlBWHdEOztBQUFBLHlCQVk1Q04sNENBQUssQ0FBQ1YsUUFBTixDQUFlOUMsWUFBWSxDQUFDWSxPQUE1QixDQVo0QztBQUFBO0FBQUEsTUFZbkVBLE9BWm1FO0FBQUEsTUFZMURtRCxVQVowRDs7QUFBQSx5QkFhckJQLDRDQUFLLENBQUNWLFFBQU4sQ0FDbkQ5QyxZQUFZLENBQUNhLG1CQURzQyxDQWJxQjtBQUFBO0FBQUEsTUFhbkVBLG1CQWJtRTtBQUFBLE1BYTlDbUQscUJBYjhDOztBQUFBLHlCQWdCeEJSLDRDQUFLLENBQUNWLFFBQU4sQ0FDaEQ5QyxZQUFZLENBQUNjLGlCQURtQyxDQWhCd0I7QUFBQTtBQUFBLE1BZ0JuRUEsaUJBaEJtRTtBQUFBLE1BZ0JoRG1ELG9CQWhCZ0Q7O0FBQUEsMEJBbUJ0Q1QsNENBQUssQ0FBQ1YsUUFBTixDQUFlOUMsWUFBWSxDQUFDZ0IsVUFBNUIsQ0FuQnNDO0FBQUE7QUFBQSxNQW1CbkVBLFVBbkJtRTtBQUFBLE1BbUJ2RGtELGFBbkJ1RDs7QUFBQSwwQkFvQnBDViw0Q0FBSyxDQUFDVixRQUFOLENBQ3BDOUMsWUFBWSxDQUFDa0IsV0FEdUIsQ0FwQm9DO0FBQUE7QUFBQSxNQW9CbkVBLFdBcEJtRTtBQUFBLE1Bb0J0RGlELGNBcEJzRDs7QUFBQSxtQkF3QmhDckIsc0RBQVEsQ0FDaEQ5QyxZQUFZLENBQUNtQixhQURtQyxDQXhCd0I7QUFBQSxNQXdCbkVBLGFBeEJtRTtBQUFBLE1Bd0JwRGlELGdCQXhCb0Q7O0FBQUEsbUJBMkJ0Q3RCLHNEQUFRLENBQVUsS0FBVixDQTNCOEI7QUFBQSxNQTJCbkV6QixVQTNCbUU7QUFBQSxNQTJCdkRnRCxhQTNCdUQ7O0FBQUEsb0JBNEJ4Q3ZCLHNEQUFRLENBQWlCO0FBQ3pEdEIsU0FBSyxFQUFFLEVBRGtEO0FBRXpEQyxRQUFJLEVBQUVDLEdBRm1EO0FBR3pEQyxRQUFJLEVBQUVELEdBSG1EO0FBSXpERSxTQUFLLEVBQUUsRUFKa0Q7QUFLekRDLFFBQUksRUFBRUgsR0FMbUQ7QUFNekRJLFFBQUksRUFBRUosR0FObUQ7QUFPekRLLFNBQUssRUFBRSxFQVBrRDtBQVF6REMsUUFBSSxFQUFFTixHQVJtRDtBQVN6RE8sUUFBSSxFQUFFUCxHQVRtRDtBQVV6RFEsWUFBUSxFQUFFLEVBVitDO0FBV3pEQyxXQUFPLEVBQUU7QUFYZ0QsR0FBakIsQ0E1QmdDO0FBQUEsTUE0Qm5FbUMsU0E1Qm1FO0FBQUEsTUE0QnhEQyxZQTVCd0Q7O0FBQUEsMEJBMENqQmYsNENBQUssQ0FBQ1YsUUFBTixDQUV2RGxDLE9BQU8sR0FBR0EsT0FBSCxHQUFhLEVBRm1DLENBMUNpQjtBQUFBO0FBQUEsTUEwQ25FNEQsb0JBMUNtRTtBQUFBLE1BMEM3Q0Msd0JBMUM2Qzs7QUFBQSxvQkE2QzdDM0Isc0RBQVEsQ0FBVSxLQUFWLENBN0NxQztBQUFBLE1BNkNuRTRCLE1BN0NtRTtBQUFBLE1BNkMzREMsVUE3QzJEOztBQStDMUUsTUFBTUMsK0JBQStCLEdBQUcsU0FBbENBLCtCQUFrQyxDQUN0Q3BFLEtBRHNDLEVBRXRDcUUsR0FGc0MsRUFHdENDLEVBSHNDLEVBSW5DO0FBQ0gsUUFBTUMsSUFBSSxHQUFHLDZGQUFJbkUsT0FBUCxDQUFWLENBREcsQ0FFSDtBQUNBO0FBQ0E7OztBQUNBLFFBQUlvRSxZQUF5QixHQUFHcEUsT0FBTyxDQUFDcUUsTUFBUixDQUM5QixVQUFDQyxJQUFEO0FBQUEsYUFBdUJBLElBQUksQ0FBQ0osRUFBTCxLQUFZQSxFQUFuQztBQUFBLEtBRDhCLEVBRTlCLENBRjhCLENBQWhDOztBQUdBLFFBQUksQ0FBQ0UsWUFBTCxFQUFtQjtBQUNqQkEsa0JBQVksR0FBR1Isb0JBQW9CLENBQUNTLE1BQXJCLENBQ2IsVUFBQ0MsSUFBRDtBQUFBLGVBQXVCQSxJQUFJLENBQUNKLEVBQUwsS0FBWUEsRUFBbkM7QUFBQSxPQURhLEVBRWIsQ0FGYSxDQUFmO0FBR0Q7O0FBRUQsUUFBTUssYUFBcUIsR0FBR0osSUFBSSxDQUFDSyxPQUFMLENBQWFKLFlBQWIsQ0FBOUI7QUFDQUEsZ0JBQVksQ0FBQ0gsR0FBRCxDQUFaLEdBQW9CckUsS0FBcEI7QUFDQXVFLFFBQUksQ0FBQ0ksYUFBRCxDQUFKLEdBQXNCSCxZQUF0QjtBQUNBakIsY0FBVSxDQUFDZ0IsSUFBRCxDQUFWO0FBQ0QsR0F0QkQ7O0FBL0MwRSxvQkF1RVBqQyxzREFBUSxDQUN6RTlDLFlBQVksQ0FBQ29DLHlCQUQ0RCxDQXZFRDtBQUFBLE1BdUVuRUEseUJBdkVtRTtBQUFBLE1BdUV4Q2lELDZCQXZFd0M7O0FBMkUxRSxTQUNFLE1BQUMsUUFBRDtBQUNFLFNBQUssRUFBRTtBQUNMcEYsVUFBSSxFQUFKQSxJQURLO0FBRUw4QyxhQUFPLEVBQVBBLE9BRks7QUFHTDNDLGVBQVMsRUFBVEEsU0FISztBQUlMNEMsa0JBQVksRUFBWkEsWUFKSztBQUtMM0MsV0FBSyxFQUFMQSxLQUxLO0FBTUw0QyxjQUFRLEVBQVJBLFFBTks7QUFPTEMsMkJBQXFCLEVBQXJCQSxxQkFQSztBQVFMQyw4QkFBd0IsRUFBeEJBLHdCQVJLO0FBU0w3QyxxQkFBZSxFQUFmQSxlQVRLO0FBVUw4Qyx3QkFBa0IsRUFBbEJBLGtCQVZLO0FBV0x6QyxrQkFBWSxFQUFaQSxZQVhLO0FBWUwwQyxnQkFBVSxFQUFWQSxVQVpLO0FBYUxDLHdCQUFrQixFQUFsQkEsa0JBYks7QUFjTEMsMkJBQXFCLEVBQXJCQSxxQkFkSztBQWVMTSxlQUFTLEVBQVRBLFNBZks7QUFlTUMsa0JBQVksRUFBWkEsWUFmTjtBQWdCTEwsdUJBQWlCLEVBQWpCQSxpQkFoQks7QUFpQkxDLDBCQUFvQixFQUFwQkEsb0JBakJLO0FBa0JMa0IscUNBQStCLEVBQS9CQSwrQkFsQks7QUFtQkxoRSxhQUFPLEVBQVBBLE9BbkJLO0FBb0JMbUQsZ0JBQVUsRUFBVkEsVUFwQks7QUFxQkxsRCx5QkFBbUIsRUFBbkJBLG1CQXJCSztBQXNCTG1ELDJCQUFxQixFQUFyQkEscUJBdEJLO0FBdUJMbEQsdUJBQWlCLEVBQWpCQSxpQkF2Qks7QUF3QkxtRCwwQkFBb0IsRUFBcEJBLG9CQXhCSztBQXlCTGpELGdCQUFVLEVBQVZBLFVBekJLO0FBMEJMa0QsbUJBQWEsRUFBYkEsYUExQks7QUEyQkxoRCxpQkFBVyxFQUFYQSxXQTNCSztBQTRCTGlELG9CQUFjLEVBQWRBLGNBNUJLO0FBNkJMaEQsbUJBQWEsRUFBYkEsYUE3Qks7QUE4QkxpRCxzQkFBZ0IsRUFBaEJBLGdCQTlCSztBQStCTC9DLGdCQUFVLEVBQVZBLFVBL0JLO0FBZ0NMZ0QsbUJBQWEsRUFBYkEsYUFoQ0s7QUFpQ0xDLGVBQVMsRUFBVEEsU0FqQ0s7QUFrQ0xDLGtCQUFZLEVBQVpBLFlBbENLO0FBbUNMQywwQkFBb0IsRUFBcEJBLG9CQW5DSztBQW9DTEMsOEJBQXdCLEVBQXhCQSx3QkFwQ0s7QUFxQ0xyQywrQkFBeUIsRUFBekJBLHlCQXJDSztBQXNDTGlELG1DQUE2QixFQUE3QkEsNkJBdENLO0FBdUNMWCxZQUFNLEVBQU5BLE1BdkNLO0FBd0NMQyxnQkFBVSxFQUFWQTtBQXhDSyxLQURUO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0E0Q0c5QixRQTVDSCxDQURGO0FBZ0RELENBM0hEOztHQUFNRCxxQjs7S0FBQUEscUI7QUE2SE4iLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvX2FwcC40YWZkMmU1ZDI4N2U2MTBkZTAwNC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0LCB7IGNyZWF0ZUNvbnRleHQsIHVzZVN0YXRlLCBSZWFjdEVsZW1lbnQgfSBmcm9tICdyZWFjdCc7XG5pbXBvcnQgeyB2NCBhcyB1dWlkdjQgfSBmcm9tICd1dWlkJztcblxuaW1wb3J0IHtcbiAgc2l6ZXMsXG4gIHZpZXdQb3NpdGlvbnMsXG4gIHBsb3RzUHJvcG9ydGlvbnNPcHRpb25zLFxufSBmcm9tICcuLi9jb21wb25lbnRzL2NvbnN0YW50cyc7XG5pbXBvcnQge1xuICBTaXplUHJvcHMsXG4gIFBsb3RQcm9wcyxcbiAgVHJpcGxlUHJvcHMsXG4gIEN1c3RvbWl6ZVByb3BzLFxufSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XG5pbXBvcnQgeyBvdmVybGF5T3B0aW9ucyB9IGZyb20gJy4uL2NvbXBvbmVudHMvY29uc3RhbnRzJztcbmltcG9ydCB7IHdvcmtzcGFjZXMgfSBmcm9tICcuLi93b3Jrc3BhY2VzL29mZmxpbmUnO1xuXG5leHBvcnQgaW50ZXJmYWNlIExlZnRTaWRlU3RhdGVQcm92aWRlclByb3BzIHtcbiAgY2hpbGRyZW46IFJlYWN0RWxlbWVudDtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBMZWZ0U2lkZVN0YXRlIHtcbiAgc2l6ZTogU2l6ZVByb3BzO1xuICBub3JtYWxpemU6IGJvb2xlYW47XG4gIHN0YXRzOiBib29sZWFuO1xuICBvdmVybGF5UG9zaXRpb246IHN0cmluZztcbiAgb3ZlcmxheTogUGxvdFByb3BzW107XG4gIHRyaXBsZXM6IFRyaXBsZVByb3BzW107XG4gIG92ZXJsYXlQbG90czogVHJpcGxlUHJvcHNbXTtcbiAgd29ya3NwYWNlRm9sZGVyczogc3RyaW5nW107XG4gIG9wZW5PdmVybGF5RGF0YU1lbnU6IGJvb2xlYW47XG4gIHZpZXdQbG90c1Bvc2l0aW9uOiBib29sZWFuO1xuICBsdW1pc2VjdGlvbjogc3RyaW5nIHwgbnVtYmVyO1xuICByaWdodFNpZGVTaXplOiBTaXplUHJvcHM7XG4gIEpTUk9PVG1vZGU6IGJvb2xlYW47XG4gIGN1c3RvbWl6ZVByb3BzOiBDdXN0b21pemVQcm9wcztcbiAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbjogbnVtYmVyO1xufVxuXG5leHBvcnQgY29uc3QgaW5pdGlhbFN0YXRlOiBhbnkgPSB7XG4gIHNpemU6IHNpemVzLm1lZGl1bS5zaXplLFxuICBub3JtYWxpemU6ICdUcnVlJyxcbiAgc3RhdHM6IHRydWUsXG4gIG92ZXJsYXlQb3NpdGlvbjogb3ZlcmxheU9wdGlvbnNbMF0udmFsdWUsXG4gIG92ZXJsYXk6IHVuZGVmaW5lZCxcbiAgb3ZlcmxheVBsb3RzOiBbXSxcbiAgdHJpcGxlczogW10sXG4gIG9wZW5PdmVybGF5RGF0YU1lbnU6IGZhbHNlLFxuICB2aWV3UGxvdHNQb3NpdGlvbjogdmlld1Bvc2l0aW9uc1sxXS52YWx1ZSxcbiAgcHJvcG9ydGlvbjogcGxvdHNQcm9wb3J0aW9uc09wdGlvbnNbMF0udmFsdWUsXG4gIGx1bWlzZWN0aW9uOiAtMSxcbiAgcmlnaHRTaWRlU2l6ZTogc2l6ZXMuZmlsbC5zaXplLFxuICBKU1JPT1Rtb2RlOiBmYWxzZSxcbiAgc2hvcnRjdXRzOiBbXSxcbiAgY3VzdG9taXplUHJvcHM6IHtcbiAgICB4dHlwZTogJycsXG4gICAgeG1pbjogTmFOLFxuICAgIHhtYXg6IE5hTixcbiAgICB5dHlwZTogJycsXG4gICAgeW1pbjogTmFOLFxuICAgIHltYXg6IE5hTixcbiAgICB6dHlwZTogJycsXG4gICAgem1pbjogTmFOLFxuICAgIHptYXg6IE5hTixcbiAgICBkcmF3b3B0czogJycsXG4gICAgd2l0aHJlZjogJycsXG4gIH0sXG4gIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW46IE1hdGgucm91bmQobmV3IERhdGUoKS5nZXRUaW1lKCkgLyAxMDAwMCkgKiAxMCxcbn07XG5cbmV4cG9ydCBpbnRlcmZhY2UgQWN0aW9uUHJvcHMge1xuICB0eXBlOiBzdHJpbmc7XG4gIHBheWxvYWQ6IGFueTtcbn1cblxuY29uc3Qgc3RvcmUgPSBjcmVhdGVDb250ZXh0KGluaXRpYWxTdGF0ZSk7XG5jb25zdCB7IFByb3ZpZGVyIH0gPSBzdG9yZTtcblxuY29uc3QgTGVmdFNpZGVTdGF0ZVByb3ZpZGVyID0gKHsgY2hpbGRyZW4gfTogTGVmdFNpZGVTdGF0ZVByb3ZpZGVyUHJvcHMpID0+IHtcbiAgY29uc3QgW3NpemUsIHNldFNpemVdID0gdXNlU3RhdGU8bnVtYmVyPihpbml0aWFsU3RhdGUuc2l6ZSk7XG4gIGNvbnN0IFtub3JtYWxpemUsIHNldE5vcm1hbGl6ZV0gPSB1c2VTdGF0ZTxib29sZWFuPihpbml0aWFsU3RhdGUubm9ybWFsaXplKTtcbiAgY29uc3QgW3N0YXRzLCBzZXRTdGF0c10gPSB1c2VTdGF0ZTxib29sZWFuPihpbml0aWFsU3RhdGUuc3RhdHMpO1xuICBjb25zdCBbcGxvdHNXaGljaEFyZU92ZXJsYWlkLCBzZXRQbG90c1doaWNoQXJlT3ZlcmxhaWRdID0gdXNlU3RhdGUoe30pO1xuICBjb25zdCBbb3ZlcmxheVBvc2l0aW9uLCBzZXRPdmVybGFpUG9zaXRpb25dID0gdXNlU3RhdGUoXG4gICAgaW5pdGlhbFN0YXRlLm92ZXJsYXlQb3NpdGlvblxuICApO1xuICBjb25zdCBbb3ZlcmxheVBsb3RzLCBzZXRPdmVybGF5XSA9IHVzZVN0YXRlKGluaXRpYWxTdGF0ZS5vdmVybGF5UGxvdHMpO1xuICBjb25zdCBbaW1hZ2VSZWZTY3JvbGxEb3duLCBzZXRJbWFnZVJlZlNjcm9sbERvd25dID0gdXNlU3RhdGUobnVsbCk7XG4gIGNvbnN0IFtwbG90U2VhcmNoRm9sZGVycywgc2V0UGxvdFNlYXJjaEZvbGRlcnNdID0gUmVhY3QudXNlU3RhdGUoW10pO1xuICBjb25zdCBbd29ya3NwYWNlLCBzZXRXb3Jrc3BhY2VdID0gUmVhY3QudXNlU3RhdGUod29ya3NwYWNlc1swXS53b3Jrc3BhY2VzWzFdLmxhYmVsKTtcbiAgY29uc3QgW3RyaXBsZXMsIHNldFRyaXBsZXNdID0gUmVhY3QudXNlU3RhdGUoaW5pdGlhbFN0YXRlLnRyaXBsZXMpO1xuICBjb25zdCBbb3Blbk92ZXJsYXlEYXRhTWVudSwgdG9nZ2xlT3ZlcmxheURhdGFNZW51XSA9IFJlYWN0LnVzZVN0YXRlKFxuICAgIGluaXRpYWxTdGF0ZS5vcGVuT3ZlcmxheURhdGFNZW51XG4gICk7XG4gIGNvbnN0IFt2aWV3UGxvdHNQb3NpdGlvbiwgc2V0Vmlld1Bsb3RzUG9zaXRpb25dID0gUmVhY3QudXNlU3RhdGUoXG4gICAgaW5pdGlhbFN0YXRlLnZpZXdQbG90c1Bvc2l0aW9uXG4gICk7XG4gIGNvbnN0IFtwcm9wb3J0aW9uLCBzZXRQcm9wb3J0aW9uXSA9IFJlYWN0LnVzZVN0YXRlKGluaXRpYWxTdGF0ZS5wcm9wb3J0aW9uKTtcbiAgY29uc3QgW2x1bWlzZWN0aW9uLCBzZXRMdW1pc2VjdGlvbl0gPSBSZWFjdC51c2VTdGF0ZShcbiAgICBpbml0aWFsU3RhdGUubHVtaXNlY3Rpb25cbiAgKTtcblxuICBjb25zdCBbcmlnaHRTaWRlU2l6ZSwgc2V0UmlnaHRTaWRlU2l6ZV0gPSB1c2VTdGF0ZTxudW1iZXI+KFxuICAgIGluaXRpYWxTdGF0ZS5yaWdodFNpZGVTaXplXG4gICk7XG4gIGNvbnN0IFtKU1JPT1Rtb2RlLCBzZXRKU1JPT1Rtb2RlXSA9IHVzZVN0YXRlPGJvb2xlYW4+KGZhbHNlKTtcbiAgY29uc3QgW2N1c3RvbWl6ZSwgc2V0Q3VzdG9taXplXSA9IHVzZVN0YXRlPEN1c3RvbWl6ZVByb3BzPih7XG4gICAgeHR5cGU6ICcnLFxuICAgIHhtaW46IE5hTixcbiAgICB4bWF4OiBOYU4sXG4gICAgeXR5cGU6ICcnLFxuICAgIHltaW46IE5hTixcbiAgICB5bWF4OiBOYU4sXG4gICAgenR5cGU6ICcnLFxuICAgIHptaW46IE5hTixcbiAgICB6bWF4OiBOYU4sXG4gICAgZHJhd29wdHM6ICcnLFxuICAgIHdpdGhyZWY6ICcnLFxuICB9KTtcblxuICBjb25zdCBbcnVuc19zZXRfZm9yX292ZXJsYXksIHNldF9ydW5zX3NldF9mb3Jfb3ZlcmxheV0gPSBSZWFjdC51c2VTdGF0ZTxcbiAgICBUcmlwbGVQcm9wc1tdXG4gID4odHJpcGxlcyA/IHRyaXBsZXMgOiBbXSk7XG4gIGNvbnN0IFt1cGRhdGUsIHNldF91cGRhdGVdID0gdXNlU3RhdGU8Ym9vbGVhbj4oZmFsc2UpO1xuXG4gIGNvbnN0IGNoYW5nZV92YWx1ZV9pbl9yZWZlcmVuY2VfdGFibGUgPSAoXG4gICAgdmFsdWU6IHN0cmluZyB8IG51bWJlcixcbiAgICBrZXk6IHN0cmluZyxcbiAgICBpZDogc3RyaW5nIHwgbnVtYmVyIHwgYm9vbGVhblxuICApID0+IHtcbiAgICBjb25zdCBjb3B5ID0gWy4uLnRyaXBsZXNdO1xuICAgIC8vdHJpcGxlcyBhcmUgdGhvc2UgcnVucyB3aGljaCBhcmUgYWxyZWFkeSBvdmVybGFpZC5cbiAgICAvL3J1bnNfc2V0X2Zvcl9vdmVybGF5IGFyZSBydW5zIHdoaWNoIGFyZSBzZWtlY3RlZCBmb3Igb3ZlcmxheSxcbiAgICAvL2J1dCBub3Qgb3ZlcmxhaWQgeWV0LlxuICAgIGxldCBjdXJyZW50X2xpbmU6IFRyaXBsZVByb3BzID0gdHJpcGxlcy5maWx0ZXIoXG4gICAgICAobGluZTogVHJpcGxlUHJvcHMpID0+IGxpbmUuaWQgPT09IGlkXG4gICAgKVswXTtcbiAgICBpZiAoIWN1cnJlbnRfbGluZSkge1xuICAgICAgY3VycmVudF9saW5lID0gcnVuc19zZXRfZm9yX292ZXJsYXkuZmlsdGVyKFxuICAgICAgICAobGluZTogVHJpcGxlUHJvcHMpID0+IGxpbmUuaWQgPT09IGlkXG4gICAgICApWzBdO1xuICAgIH1cblxuICAgIGNvbnN0IGluZGV4X29mX2xpbmU6IG51bWJlciA9IGNvcHkuaW5kZXhPZihjdXJyZW50X2xpbmUpO1xuICAgIGN1cnJlbnRfbGluZVtrZXldID0gdmFsdWU7XG4gICAgY29weVtpbmRleF9vZl9saW5lXSA9IGN1cnJlbnRfbGluZTtcbiAgICBzZXRUcmlwbGVzKGNvcHkpO1xuICB9O1xuXG4gIGNvbnN0IFt1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuLCBzZXRfdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbl0gPSB1c2VTdGF0ZShcbiAgICBpbml0aWFsU3RhdGUudXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhblxuICApO1xuXG4gIHJldHVybiAoXG4gICAgPFByb3ZpZGVyXG4gICAgICB2YWx1ZT17e1xuICAgICAgICBzaXplLFxuICAgICAgICBzZXRTaXplLFxuICAgICAgICBub3JtYWxpemUsXG4gICAgICAgIHNldE5vcm1hbGl6ZSxcbiAgICAgICAgc3RhdHMsXG4gICAgICAgIHNldFN0YXRzLFxuICAgICAgICBwbG90c1doaWNoQXJlT3ZlcmxhaWQsXG4gICAgICAgIHNldFBsb3RzV2hpY2hBcmVPdmVybGFpZCxcbiAgICAgICAgb3ZlcmxheVBvc2l0aW9uLFxuICAgICAgICBzZXRPdmVybGFpUG9zaXRpb24sXG4gICAgICAgIG92ZXJsYXlQbG90cyxcbiAgICAgICAgc2V0T3ZlcmxheSxcbiAgICAgICAgaW1hZ2VSZWZTY3JvbGxEb3duLFxuICAgICAgICBzZXRJbWFnZVJlZlNjcm9sbERvd24sXG4gICAgICAgIHdvcmtzcGFjZSwgc2V0V29ya3NwYWNlLFxuICAgICAgICBwbG90U2VhcmNoRm9sZGVycyxcbiAgICAgICAgc2V0UGxvdFNlYXJjaEZvbGRlcnMsXG4gICAgICAgIGNoYW5nZV92YWx1ZV9pbl9yZWZlcmVuY2VfdGFibGUsXG4gICAgICAgIHRyaXBsZXMsXG4gICAgICAgIHNldFRyaXBsZXMsXG4gICAgICAgIG9wZW5PdmVybGF5RGF0YU1lbnUsXG4gICAgICAgIHRvZ2dsZU92ZXJsYXlEYXRhTWVudSxcbiAgICAgICAgdmlld1Bsb3RzUG9zaXRpb24sXG4gICAgICAgIHNldFZpZXdQbG90c1Bvc2l0aW9uLFxuICAgICAgICBwcm9wb3J0aW9uLFxuICAgICAgICBzZXRQcm9wb3J0aW9uLFxuICAgICAgICBsdW1pc2VjdGlvbixcbiAgICAgICAgc2V0THVtaXNlY3Rpb24sXG4gICAgICAgIHJpZ2h0U2lkZVNpemUsXG4gICAgICAgIHNldFJpZ2h0U2lkZVNpemUsXG4gICAgICAgIEpTUk9PVG1vZGUsXG4gICAgICAgIHNldEpTUk9PVG1vZGUsXG4gICAgICAgIGN1c3RvbWl6ZSxcbiAgICAgICAgc2V0Q3VzdG9taXplLFxuICAgICAgICBydW5zX3NldF9mb3Jfb3ZlcmxheSxcbiAgICAgICAgc2V0X3J1bnNfc2V0X2Zvcl9vdmVybGF5LFxuICAgICAgICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuLFxuICAgICAgICBzZXRfdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbixcbiAgICAgICAgdXBkYXRlLFxuICAgICAgICBzZXRfdXBkYXRlLFxuICAgICAgfX1cbiAgICA+XG4gICAgICB7Y2hpbGRyZW59XG4gICAgPC9Qcm92aWRlcj5cbiAgKTtcbn07XG5cbmV4cG9ydCB7IHN0b3JlLCBMZWZ0U2lkZVN0YXRlUHJvdmlkZXIgfTtcbiJdLCJzb3VyY2VSb290IjoiIn0=