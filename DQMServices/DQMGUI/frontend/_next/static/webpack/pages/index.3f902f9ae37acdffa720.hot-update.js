webpackHotUpdate_N_E("pages/index",{

/***/ "./components/plots/zoomedPlots/zoomedOverlayPlots/zoomedOverlaidJSROOTPlot.tsx":
/*!**************************************************************************************!*\
  !*** ./components/plots/zoomedPlots/zoomedOverlayPlots/zoomedOverlaidJSROOTPlot.tsx ***!
  \**************************************************************************************/
/*! exports provided: ZoomedOverlaidJSROOTPlot */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ZoomedOverlaidJSROOTPlot", function() { return ZoomedOverlaidJSROOTPlot; });
/* harmony import */ var _babel_runtime_helpers_esm_defineProperty__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/defineProperty */ "./node_modules/@babel/runtime/helpers/esm/defineProperty.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/regenerator */ "./node_modules/@babel/runtime/regenerator/index.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @babel/runtime/helpers/esm/asyncToGenerator */ "./node_modules/@babel/runtime/helpers/esm/asyncToGenerator.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var clean_deep__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! clean-deep */ "./node_modules/clean-deep/src/index.js");
/* harmony import */ var clean_deep__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(clean_deep__WEBPACK_IMPORTED_MODULE_4__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_5___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_5__);
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../../../config/config */ "./config/config.ts");
/* harmony import */ var _hooks_useRequest__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../../../hooks/useRequest */ "./hooks/useRequest.tsx");
/* harmony import */ var _containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../../../../containers/display/styledComponents */ "./containers/display/styledComponents.tsx");
/* harmony import */ var _plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../../plot/singlePlot/utils */ "./components/plots/plot/singlePlot/utils.ts");
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_11__ = __webpack_require__(/*! ../../../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_12__ = __webpack_require__(/*! ../../../../hooks/useBlinkOnUpdate */ "./hooks/useBlinkOnUpdate.tsx");




var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/plots/zoomedPlots/zoomedOverlayPlots/zoomedOverlaidJSROOTPlot.tsx",
    _s2 = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_3___default.a.createElement;

function ownKeys(object, enumerableOnly) { var keys = Object.keys(object); if (Object.getOwnPropertySymbols) { var symbols = Object.getOwnPropertySymbols(object); if (enumerableOnly) symbols = symbols.filter(function (sym) { return Object.getOwnPropertyDescriptor(object, sym).enumerable; }); keys.push.apply(keys, symbols); } return keys; }

function _objectSpread(target) { for (var i = 1; i < arguments.length; i++) { var source = arguments[i] != null ? arguments[i] : {}; if (i % 2) { ownKeys(Object(source), true).forEach(function (key) { Object(_babel_runtime_helpers_esm_defineProperty__WEBPACK_IMPORTED_MODULE_0__["default"])(target, key, source[key]); }); } else if (Object.getOwnPropertyDescriptors) { Object.defineProperties(target, Object.getOwnPropertyDescriptors(source)); } else { ownKeys(Object(source)).forEach(function (key) { Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key)); }); } } return target; }













var drawJSROOT = /*#__PURE__*/function () {
  var _ref = Object(_babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_2__["default"])( /*#__PURE__*/_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_1___default.a.mark(function _callee(histogramParam, id, overlaidJSROOTPlot) {
    return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_1___default.a.wrap(function _callee$(_context) {
      while (1) {
        switch (_context.prev = _context.next) {
          case 0:
            _context.next = 2;
            return JSROOT.cleanup("".concat(histogramParam).concat(id));

          case 2:
            //@ts-ignore
            JSROOT.draw("".concat(histogramParam).concat(id), //@ts-ignore
            JSROOT.parse(JSON.stringify(overlaidJSROOTPlot)), "".concat(histogramParam));

          case 3:
          case "end":
            return _context.stop();
        }
      }
    }, _callee);
  }));

  return function drawJSROOT(_x, _x2, _x3) {
    return _ref.apply(this, arguments);
  };
}();

var ZoomedOverlaidJSROOTPlot = function ZoomedOverlaidJSROOTPlot(_ref2) {
  _s2();

  var _s = $RefreshSig$(),
      _params_for_api$width;

  var selected_plot = _ref2.selected_plot,
      params_for_api = _ref2.params_for_api,
      id = _ref2.id;
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_5__["useRouter"])();
  var query = router.query;

  var _useRequest = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_7__["useRequest"])(Object(_config_config__WEBPACK_IMPORTED_MODULE_6__["get_jroot_plot"])(params_for_api), {}, [selected_plot.name]),
      data = _useRequest.data;

  var overlaid_plots_runs_and_datasets = (params_for_api === null || params_for_api === void 0 ? void 0 : params_for_api.overlay_plot) ? params_for_api.overlay_plot.map(_s(function (plot) {
    _s();

    var copy = _objectSpread({}, params_for_api);

    if (plot.dataset_name) {
      copy.dataset_name = plot.dataset_name;
    }

    copy.run_number = plot.run_number;

    var _useRequest2 = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_7__["useRequest"])(Object(_config_config__WEBPACK_IMPORTED_MODULE_6__["get_jroot_plot"])(copy), {}, [selected_plot.name, query.lumi]),
        data = _useRequest2.data;

    return data;
  }, "1yL0HPIyJepm4RzqC786FzG3yLI=", false, function () {
    return [_hooks_useRequest__WEBPACK_IMPORTED_MODULE_7__["useRequest"]];
  })) : [];
  overlaid_plots_runs_and_datasets.push(data);
  var overlaidJSROOTPlot = {}; //checking how many histograms are overlaid, because just separated objects
  // (i.e separate variables ) to JSROOT.CreateTHStack() func

  if (overlaid_plots_runs_and_datasets.length === 0) {
    return null;
  } else if (overlaid_plots_runs_and_datasets.length === 1) {
    var histogram1 = overlaid_plots_runs_and_datasets[0]; //@ts-ignore

    overlaidJSROOTPlot = JSROOT.CreateTHStack(histogram1);
  } else if (overlaid_plots_runs_and_datasets.length === 2) {
    var _histogram = overlaid_plots_runs_and_datasets[0];
    var histogram2 = overlaid_plots_runs_and_datasets[1]; //@ts-ignore

    overlaidJSROOTPlot = JSROOT.CreateTHStack(_histogram, histogram2);
  } else if (overlaid_plots_runs_and_datasets.length === 3) {
    var _histogram2 = overlaid_plots_runs_and_datasets[0];
    var _histogram3 = overlaid_plots_runs_and_datasets[1];
    var histogram3 = overlaid_plots_runs_and_datasets[2]; //@ts-ignore

    overlaidJSROOTPlot = JSROOT.CreateTHStack(_histogram2, _histogram3, histogram3);
  } else if (overlaid_plots_runs_and_datasets.length === 4) {
    var _histogram4 = overlaid_plots_runs_and_datasets[0];
    var _histogram5 = overlaid_plots_runs_and_datasets[1];
    var _histogram6 = overlaid_plots_runs_and_datasets[2];
    var histogram4 = overlaid_plots_runs_and_datasets[3]; //@ts-ignore

    overlaidJSROOTPlot = JSROOT.CreateTHStack(_histogram4, _histogram5, _histogram6, histogram4);
  }

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_3___default.a.useContext(_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_11__["store"]),
      updated_by_not_older_than = _React$useContext.updated_by_not_older_than;

  var histogramParam = params_for_api.normalize ? 'hist' : 'nostack'; //make sure that no null histograms are passed to draw func.
  //on first, second reneder overlaidJSROOTPlot.fHists.arr is [null, null]
  //@ts-ignore

  Object(react__WEBPACK_IMPORTED_MODULE_3__["useEffect"])(function () {
    if (clean_deep__WEBPACK_IMPORTED_MODULE_4___default()(overlaidJSROOTPlot.fHists.arr).length === overlaidJSROOTPlot.fHists.arr.length) {
      drawJSROOT(histogramParam, id, overlaidJSROOTPlot);
    }
  }, [updated_by_not_older_than, data, params_for_api.lumi, params_for_api.overlay_plot, params_for_api.dataset_name, params_for_api.run_number, params_for_api.normalize]);

  var _useBlinkOnUpdate = Object(_hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_12__["useBlinkOnUpdate"])(),
      blink = _useBlinkOnUpdate.blink;

  return __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_8__["StyledCol"], {
    space: 2,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 143,
      columnNumber: 5
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_8__["StyledPlotRow"], {
    isLoading: blink.toString(),
    animation: (_config_config__WEBPACK_IMPORTED_MODULE_6__["functions_config"].mode === 'ONLINE').toString(),
    minheight: params_for_api.height,
    width: (_params_for_api$width = params_for_api.width) === null || _params_for_api$width === void 0 ? void 0 : _params_for_api$width.toString(),
    is_plot_selected: true.toString(),
    nopointer: true.toString(),
    justifycontent: "center",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 144,
      columnNumber: 7
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_8__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_9__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 153,
      columnNumber: 9
    }
  }, selected_plot.displayedName), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_8__["Column"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 156,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_10__["Button"], {
    type: "link",
    onClick: function onClick() {
      return Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_9__["removePlotFromRightSide"])(query, selected_plot);
    },
    icon: __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_8__["MinusIcon"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 160,
        columnNumber: 19
      }
    }),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 157,
      columnNumber: 11
    }
  })), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_8__["ImageDiv"], {
    style: {
      display: params_for_api.normalize ? '' : 'none'
    },
    id: "hist".concat(id),
    width: params_for_api.width,
    height: params_for_api.height,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 163,
      columnNumber: 9
    }
  }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_8__["ImageDiv"], {
    style: {
      display: params_for_api.normalize ? 'none' : ''
    },
    id: "nostack".concat(id),
    width: params_for_api.width,
    height: params_for_api.height,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 169,
      columnNumber: 9
    }
  })));
};

_s2(ZoomedOverlaidJSROOTPlot, "+WeUMJv6bodG3xe4maZ3ewSfUwg=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_5__["useRouter"], _hooks_useRequest__WEBPACK_IMPORTED_MODULE_7__["useRequest"], _hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_12__["useBlinkOnUpdate"]];
});

_c = ZoomedOverlaidJSROOTPlot;

var _c;

$RefreshReg$(_c, "ZoomedOverlaidJSROOTPlot");

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

/***/ "./components/plots/zoomedPlots/zoomedPlots/zoomedJSROOTPlot.tsx":
/*!***********************************************************************!*\
  !*** ./components/plots/zoomedPlots/zoomedPlots/zoomedJSROOTPlot.tsx ***!
  \***********************************************************************/
/*! exports provided: ZoomedJSROOTPlot */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ZoomedJSROOTPlot", function() { return ZoomedJSROOTPlot; });
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/regenerator */ "./node_modules/@babel/runtime/regenerator/index.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/asyncToGenerator */ "./node_modules/@babel/runtime/helpers/esm/asyncToGenerator.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../../../config/config */ "./config/config.ts");
/* harmony import */ var _hooks_useRequest__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../../../hooks/useRequest */ "./hooks/useRequest.tsx");
/* harmony import */ var _containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../../../containers/display/styledComponents */ "./containers/display/styledComponents.tsx");
/* harmony import */ var _plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../plot/singlePlot/utils */ "./components/plots/plot/singlePlot/utils.ts");
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../../../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(/*! ../../../../hooks/useBlinkOnUpdate */ "./hooks/useBlinkOnUpdate.tsx");



var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/plots/zoomedPlots/zoomedPlots/zoomedJSROOTPlot.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_2___default.a.createElement;










var drawJSROOT = /*#__PURE__*/function () {
  var _ref = Object(_babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__["default"])( /*#__PURE__*/_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.mark(function _callee(id, data) {
    return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.wrap(function _callee$(_context) {
      while (1) {
        switch (_context.prev = _context.next) {
          case 0:
            if (!document.getElementById(id)) {
              _context.next = 4;
              break;
            }

            _context.next = 3;
            return JSROOT.cleanup(id);

          case 3:
            //after cleanup we can draw a new plot
            //@ts-ignore
            JSROOT.draw(id, JSROOT.parse(JSON.stringify(data)), 'hist');

          case 4:
          case "end":
            return _context.stop();
        }
      }
    }, _callee);
  }));

  return function drawJSROOT(_x, _x2) {
    return _ref.apply(this, arguments);
  };
}();

var ZoomedJSROOTPlot = function ZoomedJSROOTPlot(_ref2) {
  _s();

  var _params_for_api$width;

  var selected_plot = _ref2.selected_plot,
      params_for_api = _ref2.params_for_api,
      id = _ref2.id;
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"])();
  var query = router.query; // const id = makeid()

  var _useRequest = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_5__["useRequest"])(Object(_config_config__WEBPACK_IMPORTED_MODULE_4__["get_jroot_plot"])(params_for_api), {}, [selected_plot.name, params_for_api.lumi]),
      data = _useRequest.data;

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_2___default.a.useContext(_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_9__["store"]),
      updated_by_not_older_than = _React$useContext.updated_by_not_older_than;

  Object(react__WEBPACK_IMPORTED_MODULE_2__["useEffect"])(function () {
    if (!!document.getElementById("".concat(id))) {
      //@ts-ignore
      drawJSROOT("".concat(id), data);
    }
  }, [data, params_for_api.lumi, updated_by_not_older_than]);

  var _useBlinkOnUpdate = Object(_hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_10__["useBlinkOnUpdate"])(),
      blink = _useBlinkOnUpdate.blink;

  return __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_6__["StyledCol"], {
    space: 2,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 69,
      columnNumber: 5
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_6__["StyledPlotRow"], {
    isLoading: blink.toString(),
    animation: (_config_config__WEBPACK_IMPORTED_MODULE_4__["functions_config"].mode === 'ONLINE').toString(),
    minheight: params_for_api.height,
    width: (_params_for_api$width = params_for_api.width) === null || _params_for_api$width === void 0 ? void 0 : _params_for_api$width.toString(),
    is_plot_selected: true.toString(),
    nopointer: true.toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 70,
      columnNumber: 7
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_6__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_7__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 78,
      columnNumber: 9
    }
  }, selected_plot.displayedName), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_6__["Column"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 81,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_8__["Button"], {
    type: "link",
    onClick: function onClick() {
      return Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_7__["removePlotFromRightSide"])(query, selected_plot);
    },
    icon: __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_6__["MinusIcon"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 85,
        columnNumber: 19
      }
    }),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 82,
      columnNumber: 11
    }
  })), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_6__["ImageDiv"], {
    id: "".concat(id),
    width: params_for_api.width,
    height: params_for_api.height,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 88,
      columnNumber: 9
    }
  })));
};

_s(ZoomedJSROOTPlot, "+WeUMJv6bodG3xe4maZ3ewSfUwg=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"], _hooks_useRequest__WEBPACK_IMPORTED_MODULE_5__["useRequest"], _hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_10__["useBlinkOnUpdate"]];
});

_c = ZoomedJSROOTPlot;

var _c;

$RefreshReg$(_c, "ZoomedJSROOTPlot");

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

/***/ "./components/utils.ts":
/*!*****************************!*\
  !*** ./components/utils.ts ***!
  \*****************************/
/*! exports provided: seperateRunAndLumiInSearch, get_label, getPathName, makeid */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "seperateRunAndLumiInSearch", function() { return seperateRunAndLumiInSearch; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "get_label", function() { return get_label; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "getPathName", function() { return getPathName; });
/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "makeid", function() { return makeid; });
var seperateRunAndLumiInSearch = function seperateRunAndLumiInSearch(runAndLumi) {
  var runAndLumiArray = runAndLumi.split(':');
  var parsedRun = runAndLumiArray[0];
  var parsedLumi = runAndLumiArray[1] ? parseInt(runAndLumiArray[1]) : 0;
  return {
    parsedRun: parsedRun,
    parsedLumi: parsedLumi
  };
};
var get_label = function get_label(info, data) {
  var value = data ? data.fString : null;

  if ((info === null || info === void 0 ? void 0 : info.type) && info.type === 'time' && value) {
    var milisec = new Date(parseInt(value) * 1000);
    var time = milisec.toUTCString();
    return time;
  } else {
    return value ? value : 'No information';
  }
};
var getPathName = function getPathName() {
  var isBrowser = function isBrowser() {
    return true;
  };

  var pathName = isBrowser() && window.location.pathname || '/';
  return pathName;
};
var makeid = function makeid() {
  var text = '';
  var possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';

  for (var i = 0; i < 5; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }

  return text;
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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy96b29tZWRPdmVybGF5UGxvdHMvem9vbWVkT3ZlcmxhaWRKU1JPT1RQbG90LnRzeCIsIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy96b29tZWRQbG90cy96b29tZWRKU1JPT1RQbG90LnRzeCIsIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy91dGlscy50cyJdLCJuYW1lcyI6WyJkcmF3SlNST09UIiwiaGlzdG9ncmFtUGFyYW0iLCJpZCIsIm92ZXJsYWlkSlNST09UUGxvdCIsIkpTUk9PVCIsImNsZWFudXAiLCJkcmF3IiwicGFyc2UiLCJKU09OIiwic3RyaW5naWZ5IiwiWm9vbWVkT3ZlcmxhaWRKU1JPT1RQbG90Iiwic2VsZWN0ZWRfcGxvdCIsInBhcmFtc19mb3JfYXBpIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJ1c2VSZXF1ZXN0IiwiZ2V0X2pyb290X3Bsb3QiLCJuYW1lIiwiZGF0YSIsIm92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzIiwib3ZlcmxheV9wbG90IiwibWFwIiwicGxvdCIsImNvcHkiLCJkYXRhc2V0X25hbWUiLCJydW5fbnVtYmVyIiwibHVtaSIsInB1c2giLCJsZW5ndGgiLCJoaXN0b2dyYW0xIiwiQ3JlYXRlVEhTdGFjayIsImhpc3RvZ3JhbTIiLCJoaXN0b2dyYW0zIiwiaGlzdG9ncmFtNCIsIlJlYWN0IiwidXNlQ29udGV4dCIsInN0b3JlIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsIm5vcm1hbGl6ZSIsInVzZUVmZmVjdCIsImNsZWFuRGVlcCIsImZIaXN0cyIsImFyciIsInVzZUJsaW5rT25VcGRhdGUiLCJibGluayIsInRvU3RyaW5nIiwiZnVuY3Rpb25zX2NvbmZpZyIsIm1vZGUiLCJoZWlnaHQiLCJ3aWR0aCIsImdldF9wbG90X2Vycm9yIiwiZGlzcGxheWVkTmFtZSIsInJlbW92ZVBsb3RGcm9tUmlnaHRTaWRlIiwiZGlzcGxheSIsImRvY3VtZW50IiwiZ2V0RWxlbWVudEJ5SWQiLCJab29tZWRKU1JPT1RQbG90Iiwic2VwZXJhdGVSdW5BbmRMdW1pSW5TZWFyY2giLCJydW5BbmRMdW1pIiwicnVuQW5kTHVtaUFycmF5Iiwic3BsaXQiLCJwYXJzZWRSdW4iLCJwYXJzZWRMdW1pIiwicGFyc2VJbnQiLCJnZXRfbGFiZWwiLCJpbmZvIiwidmFsdWUiLCJmU3RyaW5nIiwidHlwZSIsIm1pbGlzZWMiLCJEYXRlIiwidGltZSIsInRvVVRDU3RyaW5nIiwiZ2V0UGF0aE5hbWUiLCJpc0Jyb3dzZXIiLCJwYXRoTmFtZSIsIndpbmRvdyIsImxvY2F0aW9uIiwicGF0aG5hbWUiLCJtYWtlaWQiLCJ0ZXh0IiwicG9zc2libGUiLCJpIiwiY2hhckF0IiwiTWF0aCIsImZsb29yIiwicmFuZG9tIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBRUE7QUFPQTtBQUNBO0FBQ0E7QUFRQTtBQUlBO0FBQ0E7QUFDQTs7QUFPQSxJQUFNQSxVQUFVO0FBQUEsOExBQUcsaUJBQ2pCQyxjQURpQixFQUVqQkMsRUFGaUIsRUFHakJDLGtCQUhpQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxtQkFNWEMsTUFBTSxDQUFDQyxPQUFQLFdBQWtCSixjQUFsQixTQUFtQ0MsRUFBbkMsRUFOVzs7QUFBQTtBQU9qQjtBQUNBRSxrQkFBTSxDQUFDRSxJQUFQLFdBQ0tMLGNBREwsU0FDc0JDLEVBRHRCLEdBRUU7QUFDQUUsa0JBQU0sQ0FBQ0csS0FBUCxDQUFhQyxJQUFJLENBQUNDLFNBQUwsQ0FBZU4sa0JBQWYsQ0FBYixDQUhGLFlBSUtGLGNBSkw7O0FBUmlCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEdBQUg7O0FBQUEsa0JBQVZELFVBQVU7QUFBQTtBQUFBO0FBQUEsR0FBaEI7O0FBZ0JPLElB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UixFQUFiO0FBQ0EsV0FBT0QsSUFBUDtBQUNELEdBSkQsTUFJTztBQUNMLFdBQU9MLEtBQUssR0FBR0EsS0FBSCxHQUFXLGdCQUF2QjtBQUNEO0FBQ0YsQ0FWTTtBQVlBLElBQU1PLFdBQVcsR0FBRyxTQUFkQSxXQUFjLEdBQU07QUFDL0IsTUFBTUMsU0FBUyxHQUFHLFNBQVpBLFNBQVk7QUFBQTtBQUFBLEdBQWxCOztBQUNBLE1BQU1DLFFBQVEsR0FBSUQsU0FBUyxNQUFNRSxNQUFNLENBQUNDLFFBQVAsQ0FBZ0JDLFFBQWhDLElBQTZDLEdBQTlEO0FBQ0EsU0FBT0gsUUFBUDtBQUNELENBSk07QUFLQSxJQUFNSSxNQUFNLEdBQUcsU0FBVEEsTUFBUyxHQUFNO0FBQzFCLE1BQUlDLElBQUksR0FBRyxFQUFYO0FBQ0EsTUFBSUMsUUFBUSxHQUFHLHNEQUFmOztBQUVBLE9BQUssSUFBSUMsQ0FBQyxHQUFHLENBQWIsRUFBZ0JBLENBQUMsR0FBRyxDQUFwQixFQUF1QkEsQ0FBQyxFQUF4QjtBQUNFRixRQUFJLElBQUlDLFFBQVEsQ0FBQ0UsTUFBVCxDQUFnQkMsSUFBSSxDQUFDQyxLQUFMLENBQVdELElBQUksQ0FBQ0UsTUFBTCxLQUFnQkwsUUFBUSxDQUFDckQsTUFBcEMsQ0FBaEIsQ0FBUjtBQURGOztBQUdBLFNBQU9vRCxJQUFQO0FBQ0QsQ0FSTSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4zZjkwMmY5YWUzN2FjZGZmYTcyMC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IGNsZWFuRGVlcCBmcm9tICdjbGVhbi1kZWVwJztcclxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xyXG5cclxuaW1wb3J0IHsgZ2V0X2pyb290X3Bsb3QsIGZ1bmN0aW9uc19jb25maWcgfSBmcm9tICcuLi8uLi8uLi8uLi9jb25maWcvY29uZmlnJztcclxuaW1wb3J0IHtcclxuICBQYXJhbXNGb3JBcGlQcm9wcyxcclxuICBUcmlwbGVQcm9wcyxcclxuICBQbG90RGF0YVByb3BzLFxyXG4gIFF1ZXJ5UHJvcHMsXHJcbn0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyB1c2VSZXF1ZXN0IH0gZnJvbSAnLi4vLi4vLi4vLi4vaG9va3MvdXNlUmVxdWVzdCc7XHJcbmltcG9ydCB7IHVzZUVmZmVjdCB9IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHtcclxuICBTdHlsZWRDb2wsXHJcbiAgQ29sdW1uLFxyXG4gIFN0eWxlZFBsb3RSb3csXHJcbiAgUGxvdE5hbWVDb2wsXHJcbiAgTWludXNJY29uLFxyXG4gIEltYWdlRGl2LFxyXG59IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHtcclxuICByZW1vdmVQbG90RnJvbVJpZ2h0U2lkZSxcclxuICBnZXRfcGxvdF9lcnJvcixcclxufSBmcm9tICcuLi8uLi9wbG90L3NpbmdsZVBsb3QvdXRpbHMnO1xyXG5pbXBvcnQgeyBCdXR0b24gfSBmcm9tICdhbnRkJztcclxuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi8uLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnO1xyXG5pbXBvcnQgeyB1c2VCbGlua09uVXBkYXRlIH0gZnJvbSAnLi4vLi4vLi4vLi4vaG9va3MvdXNlQmxpbmtPblVwZGF0ZSc7XHJcblxyXG5pbnRlcmZhY2UgWm9vbWVkSlNST09UUGxvdHNQcm9wcyB7XHJcbiAgc2VsZWN0ZWRfcGxvdDogUGxvdERhdGFQcm9wcztcclxuICBwYXJhbXNfZm9yX2FwaTogUGFyYW1zRm9yQXBpUHJvcHM7XHJcbiAgaWQ6IHN0cmluZztcclxufVxyXG5jb25zdCBkcmF3SlNST09UID0gYXN5bmMgKFxyXG4gIGhpc3RvZ3JhbVBhcmFtOiBzdHJpbmcsXHJcbiAgaWQ6IHN0cmluZyxcclxuICBvdmVybGFpZEpTUk9PVFBsb3Q6IGFueVxyXG4pID0+IHtcclxuICAvL0B0cy1pZ25vcmVcclxuICBhd2FpdCBKU1JPT1QuY2xlYW51cChgJHtoaXN0b2dyYW1QYXJhbX0ke2lkfWApO1xyXG4gIC8vQHRzLWlnbm9yZVxyXG4gIEpTUk9PVC5kcmF3KFxyXG4gICAgYCR7aGlzdG9ncmFtUGFyYW19JHtpZH1gLFxyXG4gICAgLy9AdHMtaWdub3JlXHJcbiAgICBKU1JPT1QucGFyc2UoSlNPTi5zdHJpbmdpZnkob3ZlcmxhaWRKU1JPT1RQbG90KSksXHJcbiAgICBgJHtoaXN0b2dyYW1QYXJhbX1gXHJcbiAgKTtcclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBab29tZWRPdmVybGFpZEpTUk9PVFBsb3QgPSAoe1xyXG4gIHNlbGVjdGVkX3Bsb3QsXHJcbiAgcGFyYW1zX2Zvcl9hcGksXHJcbiAgaWQsXHJcbn06IFpvb21lZEpTUk9PVFBsb3RzUHJvcHMpID0+IHtcclxuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcclxuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcclxuXHJcbiAgY29uc3QgeyBkYXRhIH0gPSB1c2VSZXF1ZXN0KGdldF9qcm9vdF9wbG90KHBhcmFtc19mb3JfYXBpKSwge30sIFtcclxuICAgIHNlbGVjdGVkX3Bsb3QubmFtZSxcclxuICBdKTtcclxuXHJcbiAgY29uc3Qgb3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHM6IGFueVtdID0gcGFyYW1zX2Zvcl9hcGk/Lm92ZXJsYXlfcGxvdFxyXG4gICAgPyBwYXJhbXNfZm9yX2FwaS5vdmVybGF5X3Bsb3QubWFwKChwbG90OiBUcmlwbGVQcm9wcykgPT4ge1xyXG4gICAgICAgIGNvbnN0IGNvcHk6IGFueSA9IHsgLi4ucGFyYW1zX2Zvcl9hcGkgfTtcclxuXHJcbiAgICAgICAgaWYgKHBsb3QuZGF0YXNldF9uYW1lKSB7XHJcbiAgICAgICAgICBjb3B5LmRhdGFzZXRfbmFtZSA9IHBsb3QuZGF0YXNldF9uYW1lO1xyXG4gICAgICAgIH1cclxuICAgICAgICBjb3B5LnJ1bl9udW1iZXIgPSBwbG90LnJ1bl9udW1iZXI7XHJcbiAgICAgICAgY29uc3QgeyBkYXRhIH0gPSB1c2VSZXF1ZXN0KGdldF9qcm9vdF9wbG90KGNvcHkpLCB7fSwgW1xyXG4gICAgICAgICAgc2VsZWN0ZWRfcGxvdC5uYW1lLFxyXG4gICAgICAgICAgcXVlcnkubHVtaSxcclxuICAgICAgICBdKTtcclxuICAgICAgICByZXR1cm4gZGF0YTtcclxuICAgICAgfSlcclxuICAgIDogW107XHJcblxyXG4gIG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzLnB1c2goZGF0YSk7XHJcblxyXG4gIGxldCBvdmVybGFpZEpTUk9PVFBsb3Q6IGFueSA9IHt9O1xyXG5cclxuICAvL2NoZWNraW5nIGhvdyBtYW55IGhpc3RvZ3JhbXMgYXJlIG92ZXJsYWlkLCBiZWNhdXNlIGp1c3Qgc2VwYXJhdGVkIG9iamVjdHNcclxuICAvLyAoaS5lIHNlcGFyYXRlIHZhcmlhYmxlcyApIHRvIEpTUk9PVC5DcmVhdGVUSFN0YWNrKCkgZnVuY1xyXG4gIGlmIChvdmVybGFpZF9wbG90c19ydW5zX2FuZF9kYXRhc2V0cy5sZW5ndGggPT09IDApIHtcclxuICAgIHJldHVybiBudWxsO1xyXG4gIH0gZWxzZSBpZiAob3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHMubGVuZ3RoID09PSAxKSB7XHJcbiAgICBjb25zdCBoaXN0b2dyYW0xID0gb3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHNbMF07XHJcbiAgICAvL0B0cy1pZ25vcmVcclxuICAgIG92ZXJsYWlkSlNST09UUGxvdCA9IEpTUk9PVC5DcmVhdGVUSFN0YWNrKGhpc3RvZ3JhbTEpO1xyXG4gIH0gZWxzZSBpZiAob3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHMubGVuZ3RoID09PSAyKSB7XHJcbiAgICBjb25zdCBoaXN0b2dyYW0xID0gb3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHNbMF07XHJcbiAgICBjb25zdCBoaXN0b2dyYW0yID0gb3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHNbMV07XHJcbiAgICAvL0B0cy1pZ25vcmVcclxuICAgIG92ZXJsYWlkSlNST09UUGxvdCA9IEpTUk9PVC5DcmVhdGVUSFN0YWNrKGhpc3RvZ3JhbTEsIGhpc3RvZ3JhbTIpO1xyXG4gIH0gZWxzZSBpZiAob3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHMubGVuZ3RoID09PSAzKSB7XHJcbiAgICBjb25zdCBoaXN0b2dyYW0xID0gb3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHNbMF07XHJcbiAgICBjb25zdCBoaXN0b2dyYW0yID0gb3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHNbMV07XHJcbiAgICBjb25zdCBoaXN0b2dyYW0zID0gb3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHNbMl07XHJcbiAgICAvL0B0cy1pZ25vcmVcclxuICAgIG92ZXJsYWlkSlNST09UUGxvdCA9IEpTUk9PVC5DcmVhdGVUSFN0YWNrKFxyXG4gICAgICBoaXN0b2dyYW0xLFxyXG4gICAgICBoaXN0b2dyYW0yLFxyXG4gICAgICBoaXN0b2dyYW0zXHJcbiAgICApO1xyXG4gIH0gZWxzZSBpZiAob3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHMubGVuZ3RoID09PSA0KSB7XHJcbiAgICBjb25zdCBoaXN0b2dyYW0xID0gb3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHNbMF07XHJcbiAgICBjb25zdCBoaXN0b2dyYW0yID0gb3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHNbMV07XHJcbiAgICBjb25zdCBoaXN0b2dyYW0zID0gb3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHNbMl07XHJcbiAgICBjb25zdCBoaXN0b2dyYW00ID0gb3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHNbM107XHJcbiAgICAvL0B0cy1pZ25vcmVcclxuICAgIG92ZXJsYWlkSlNST09UUGxvdCA9IEpTUk9PVC5DcmVhdGVUSFN0YWNrKFxyXG4gICAgICBoaXN0b2dyYW0xLFxyXG4gICAgICBoaXN0b2dyYW0yLFxyXG4gICAgICBoaXN0b2dyYW0zLFxyXG4gICAgICBoaXN0b2dyYW00XHJcbiAgICApO1xyXG4gIH1cclxuICBjb25zdCB7IHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4gfSA9IFJlYWN0LnVzZUNvbnRleHQoc3RvcmUpO1xyXG5cclxuICBjb25zdCBoaXN0b2dyYW1QYXJhbSA9IHBhcmFtc19mb3JfYXBpLm5vcm1hbGl6ZSA/ICdoaXN0JyA6ICdub3N0YWNrJztcclxuICAvL21ha2Ugc3VyZSB0aGF0IG5vIG51bGwgaGlzdG9ncmFtcyBhcmUgcGFzc2VkIHRvIGRyYXcgZnVuYy5cclxuICAvL29uIGZpcnN0LCBzZWNvbmQgcmVuZWRlciBvdmVybGFpZEpTUk9PVFBsb3QuZkhpc3RzLmFyciBpcyBbbnVsbCwgbnVsbF1cclxuICAvL0B0cy1pZ25vcmVcclxuICB1c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgaWYgKFxyXG4gICAgICBjbGVhbkRlZXAob3ZlcmxhaWRKU1JPT1RQbG90LmZIaXN0cy5hcnIpLmxlbmd0aCA9PT1cclxuICAgICAgb3ZlcmxhaWRKU1JPT1RQbG90LmZIaXN0cy5hcnIubGVuZ3RoXHJcbiAgICApIHtcclxuICAgICAgZHJhd0pTUk9PVChoaXN0b2dyYW1QYXJhbSwgaWQsIG92ZXJsYWlkSlNST09UUGxvdCk7XHJcbiAgICB9XHJcbiAgfSwgW1xyXG4gICAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbixcclxuICAgIGRhdGEsXHJcbiAgICBwYXJhbXNfZm9yX2FwaS5sdW1pLFxyXG4gICAgcGFyYW1zX2Zvcl9hcGkub3ZlcmxheV9wbG90LFxyXG4gICAgcGFyYW1zX2Zvcl9hcGkuZGF0YXNldF9uYW1lLFxyXG4gICAgcGFyYW1zX2Zvcl9hcGkucnVuX251bWJlcixcclxuICAgIHBhcmFtc19mb3JfYXBpLm5vcm1hbGl6ZSxcclxuICBdKTtcclxuICBjb25zdCB7IGJsaW5rIH0gPSB1c2VCbGlua09uVXBkYXRlKCk7XHJcbiAgcmV0dXJuIChcclxuICAgIDxTdHlsZWRDb2wgc3BhY2U9ezJ9PlxyXG4gICAgICA8U3R5bGVkUGxvdFJvd1xyXG4gICAgICAgIGlzTG9hZGluZz17YmxpbmsudG9TdHJpbmcoKX1cclxuICAgICAgICBhbmltYXRpb249eyhmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnKS50b1N0cmluZygpfVxyXG4gICAgICAgIG1pbmhlaWdodD17cGFyYW1zX2Zvcl9hcGkuaGVpZ2h0fVxyXG4gICAgICAgIHdpZHRoPXtwYXJhbXNfZm9yX2FwaS53aWR0aD8udG9TdHJpbmcoKX1cclxuICAgICAgICBpc19wbG90X3NlbGVjdGVkPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgbm9wb2ludGVyPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgICAganVzdGlmeWNvbnRlbnQ9XCJjZW50ZXJcIlxyXG4gICAgICA+XHJcbiAgICAgICAgPFBsb3ROYW1lQ29sIGVycm9yPXtnZXRfcGxvdF9lcnJvcihzZWxlY3RlZF9wbG90KS50b1N0cmluZygpfT5cclxuICAgICAgICAgIHtzZWxlY3RlZF9wbG90LmRpc3BsYXllZE5hbWV9XHJcbiAgICAgICAgPC9QbG90TmFtZUNvbD5cclxuICAgICAgICA8Q29sdW1uPlxyXG4gICAgICAgICAgPEJ1dHRvblxyXG4gICAgICAgICAgICB0eXBlPVwibGlua1wiXHJcbiAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHJlbW92ZVBsb3RGcm9tUmlnaHRTaWRlKHF1ZXJ5LCBzZWxlY3RlZF9wbG90KX1cclxuICAgICAgICAgICAgaWNvbj17PE1pbnVzSWNvbiAvPn1cclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgPC9Db2x1bW4+XHJcbiAgICAgICAgPEltYWdlRGl2XHJcbiAgICAgICAgICBzdHlsZT17eyBkaXNwbGF5OiBwYXJhbXNfZm9yX2FwaS5ub3JtYWxpemUgPyAnJyA6ICdub25lJyB9fVxyXG4gICAgICAgICAgaWQ9e2BoaXN0JHtpZH1gfVxyXG4gICAgICAgICAgd2lkdGg9e3BhcmFtc19mb3JfYXBpLndpZHRofVxyXG4gICAgICAgICAgaGVpZ2h0PXtwYXJhbXNfZm9yX2FwaS5oZWlnaHR9XHJcbiAgICAgICAgLz5cclxuICAgICAgICA8SW1hZ2VEaXZcclxuICAgICAgICAgIHN0eWxlPXt7IGRpc3BsYXk6IHBhcmFtc19mb3JfYXBpLm5vcm1hbGl6ZSA/ICdub25lJyA6ICcnIH19XHJcbiAgICAgICAgICBpZD17YG5vc3RhY2ske2lkfWB9XHJcbiAgICAgICAgICB3aWR0aD17cGFyYW1zX2Zvcl9hcGkud2lkdGh9XHJcbiAgICAgICAgICBoZWlnaHQ9e3BhcmFtc19mb3JfYXBpLmhlaWdodH1cclxuICAgICAgICAvPlxyXG4gICAgICA8L1N0eWxlZFBsb3RSb3c+XHJcbiAgICA8L1N0eWxlZENvbD5cclxuICApO1xyXG59O1xyXG4iLCJpbXBvcnQgUmVhY3QsIHsgdXNlRWZmZWN0IH0gZnJvbSAncmVhY3QnO1xyXG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XHJcblxyXG5pbXBvcnQgeyBnZXRfanJvb3RfcGxvdCwgZnVuY3Rpb25zX2NvbmZpZyB9IGZyb20gJy4uLy4uLy4uLy4uL2NvbmZpZy9jb25maWcnO1xyXG5pbXBvcnQge1xyXG4gIFBhcmFtc0ZvckFwaVByb3BzLFxyXG4gIFBsb3REYXRhUHJvcHMsXHJcbiAgUXVlcnlQcm9wcyxcclxufSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IHVzZVJlcXVlc3QgfSBmcm9tICcuLi8uLi8uLi8uLi9ob29rcy91c2VSZXF1ZXN0JztcclxuaW1wb3J0IHtcclxuICBTdHlsZWRDb2wsXHJcbiAgU3R5bGVkUGxvdFJvdyxcclxuICBQbG90TmFtZUNvbCxcclxuICBNaW51c0ljb24sXHJcbiAgQ29sdW1uLFxyXG4gIEltYWdlRGl2LFxyXG59IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHtcclxuICByZW1vdmVQbG90RnJvbVJpZ2h0U2lkZSxcclxuICBnZXRfcGxvdF9lcnJvcixcclxufSBmcm9tICcuLi8uLi9wbG90L3NpbmdsZVBsb3QvdXRpbHMnO1xyXG5pbXBvcnQgeyBCdXR0b24gfSBmcm9tICdhbnRkJztcclxuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi8uLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnO1xyXG5pbXBvcnQgeyB1c2VCbGlua09uVXBkYXRlIH0gZnJvbSAnLi4vLi4vLi4vLi4vaG9va3MvdXNlQmxpbmtPblVwZGF0ZSc7XHJcblxyXG5pbnRlcmZhY2UgWm9vbWVkSlNST09UUGxvdHNQcm9wcyB7XHJcbiAgc2VsZWN0ZWRfcGxvdDogUGxvdERhdGFQcm9wcztcclxuICBwYXJhbXNfZm9yX2FwaTogUGFyYW1zRm9yQXBpUHJvcHM7XHJcbiAgaWQ6IHN0cmluZztcclxufVxyXG5cclxuY29uc3QgZHJhd0pTUk9PVCA9IGFzeW5jIChpZDogc3RyaW5nLCBkYXRhOiBhbnkpID0+IHtcclxuICAvL2luIG9yZGVyIHRvIGdldCBuZXcgSlNST09UIHBsb3QsIGZpcnN0IG9mIGFsbCB3ZSBuZWVkIHRvIGNsZWFuIGRpdiB3aXRoIG9sZCBwbG90XHJcbiAgaWYgKCEhZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpKSB7XHJcbiAgICAvL0B0cy1pZ25vcmVcclxuICAgIGF3YWl0IEpTUk9PVC5jbGVhbnVwKGlkKTtcclxuICAgIC8vYWZ0ZXIgY2xlYW51cCB3ZSBjYW4gZHJhdyBhIG5ldyBwbG90XHJcbiAgICAvL0B0cy1pZ25vcmVcclxuICAgIEpTUk9PVC5kcmF3KGlkLCBKU1JPT1QucGFyc2UoSlNPTi5zdHJpbmdpZnkoZGF0YSkpLCAnaGlzdCcpO1xyXG4gIH1cclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBab29tZWRKU1JPT1RQbG90ID0gKHtcclxuICBzZWxlY3RlZF9wbG90LFxyXG4gIHBhcmFtc19mb3JfYXBpLFxyXG4gIGlkLFxyXG59OiBab29tZWRKU1JPT1RQbG90c1Byb3BzKSA9PiB7XHJcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XHJcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XHJcbiAgLy8gY29uc3QgaWQgPSBtYWtlaWQoKVxyXG4gIGNvbnN0IHsgZGF0YSB9ID0gdXNlUmVxdWVzdChnZXRfanJvb3RfcGxvdChwYXJhbXNfZm9yX2FwaSksIHt9LCBbXHJcbiAgICBzZWxlY3RlZF9wbG90Lm5hbWUsXHJcbiAgICBwYXJhbXNfZm9yX2FwaS5sdW1pLFxyXG4gIF0pO1xyXG5cclxuICBjb25zdCB7IHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4gfSA9IFJlYWN0LnVzZUNvbnRleHQoc3RvcmUpO1xyXG5cclxuICB1c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgaWYgKCEhZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoYCR7aWR9YCkpIHtcclxuICAgICAgLy9AdHMtaWdub3JlXHJcbiAgICAgIGRyYXdKU1JPT1QoYCR7aWR9YCwgZGF0YSk7XHJcbiAgICB9XHJcbiAgfSwgW2RhdGEsIHBhcmFtc19mb3JfYXBpLmx1bWksIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW5dKTtcclxuXHJcbiAgY29uc3QgeyBibGluayB9ID0gdXNlQmxpbmtPblVwZGF0ZSgpO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPFN0eWxlZENvbCBzcGFjZT17Mn0+XHJcbiAgICAgIDxTdHlsZWRQbG90Um93XHJcbiAgICAgICAgaXNMb2FkaW5nPXtibGluay50b1N0cmluZygpfVxyXG4gICAgICAgIGFuaW1hdGlvbj17KGZ1bmN0aW9uc19jb25maWcubW9kZSA9PT0gJ09OTElORScpLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgbWluaGVpZ2h0PXtwYXJhbXNfZm9yX2FwaS5oZWlnaHR9XHJcbiAgICAgICAgd2lkdGg9e3BhcmFtc19mb3JfYXBpLndpZHRoPy50b1N0cmluZygpfVxyXG4gICAgICAgIGlzX3Bsb3Rfc2VsZWN0ZWQ9e3RydWUudG9TdHJpbmcoKX1cclxuICAgICAgICBub3BvaW50ZXI9e3RydWUudG9TdHJpbmcoKX1cclxuICAgICAgPlxyXG4gICAgICAgIDxQbG90TmFtZUNvbCBlcnJvcj17Z2V0X3Bsb3RfZXJyb3Ioc2VsZWN0ZWRfcGxvdCkudG9TdHJpbmcoKX0+XHJcbiAgICAgICAgICB7c2VsZWN0ZWRfcGxvdC5kaXNwbGF5ZWROYW1lfVxyXG4gICAgICAgIDwvUGxvdE5hbWVDb2w+XHJcbiAgICAgICAgPENvbHVtbj5cclxuICAgICAgICAgIDxCdXR0b25cclxuICAgICAgICAgICAgdHlwZT1cImxpbmtcIlxyXG4gICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiByZW1vdmVQbG90RnJvbVJpZ2h0U2lkZShxdWVyeSwgc2VsZWN0ZWRfcGxvdCl9XHJcbiAgICAgICAgICAgIGljb249ezxNaW51c0ljb24gLz59XHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgIDwvQ29sdW1uPlxyXG4gICAgICAgIDxJbWFnZURpdlxyXG4gICAgICAgICAgaWQ9e2Ake2lkfWB9XHJcbiAgICAgICAgICB3aWR0aD17cGFyYW1zX2Zvcl9hcGkud2lkdGh9XHJcbiAgICAgICAgICBoZWlnaHQ9e3BhcmFtc19mb3JfYXBpLmhlaWdodH1cclxuICAgICAgICAvPlxyXG4gICAgICA8L1N0eWxlZFBsb3RSb3c+XHJcbiAgICA8L1N0eWxlZENvbD5cclxuICApO1xyXG59O1xyXG4iLCJpbXBvcnQgeyBJbmZvUHJvcHMgfSBmcm9tICcuLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcblxyXG5leHBvcnQgY29uc3Qgc2VwZXJhdGVSdW5BbmRMdW1pSW5TZWFyY2ggPSAocnVuQW5kTHVtaTogc3RyaW5nKSA9PiB7XHJcbiAgY29uc3QgcnVuQW5kTHVtaUFycmF5ID0gcnVuQW5kTHVtaS5zcGxpdCgnOicpO1xyXG4gIGNvbnN0IHBhcnNlZFJ1biA9IHJ1bkFuZEx1bWlBcnJheVswXTtcclxuICBjb25zdCBwYXJzZWRMdW1pID0gcnVuQW5kTHVtaUFycmF5WzFdID8gcGFyc2VJbnQocnVuQW5kTHVtaUFycmF5WzFdKSA6IDA7XHJcblxyXG4gIHJldHVybiB7IHBhcnNlZFJ1biwgcGFyc2VkTHVtaSB9O1xyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldF9sYWJlbCA9IChpbmZvOiBJbmZvUHJvcHMsIGRhdGE/OiBhbnkpID0+IHtcclxuICBjb25zdCB2YWx1ZSA9IGRhdGEgPyBkYXRhLmZTdHJpbmcgOiBudWxsO1xyXG5cclxuICBpZiAoaW5mbz8udHlwZSAmJiBpbmZvLnR5cGUgPT09ICd0aW1lJyAmJiB2YWx1ZSkge1xyXG4gICAgY29uc3QgbWlsaXNlYyA9IG5ldyBEYXRlKHBhcnNlSW50KHZhbHVlKSAqIDEwMDApO1xyXG4gICAgY29uc3QgdGltZSA9IG1pbGlzZWMudG9VVENTdHJpbmcoKTtcclxuICAgIHJldHVybiB0aW1lO1xyXG4gIH0gZWxzZSB7XHJcbiAgICByZXR1cm4gdmFsdWUgPyB2YWx1ZSA6ICdObyBpbmZvcm1hdGlvbic7XHJcbiAgfVxyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IGdldFBhdGhOYW1lID0gKCkgPT4ge1xyXG4gIGNvbnN0IGlzQnJvd3NlciA9ICgpID0+IHR5cGVvZiB3aW5kb3cgIT09ICd1bmRlZmluZWQnO1xyXG4gIGNvbnN0IHBhdGhOYW1lID0gKGlzQnJvd3NlcigpICYmIHdpbmRvdy5sb2NhdGlvbi5wYXRobmFtZSkgfHwgJy8nO1xyXG4gIHJldHVybiBwYXRoTmFtZTtcclxufTtcclxuZXhwb3J0IGNvbnN0IG1ha2VpZCA9ICgpID0+IHtcclxuICB2YXIgdGV4dCA9ICcnO1xyXG4gIHZhciBwb3NzaWJsZSA9ICdBQkNERUZHSElKS0xNTk9QUVJTVFVWV1hZWmFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eHl6JztcclxuXHJcbiAgZm9yICh2YXIgaSA9IDA7IGkgPCA1OyBpKyspXHJcbiAgICB0ZXh0ICs9IHBvc3NpYmxlLmNoYXJBdChNYXRoLmZsb29yKE1hdGgucmFuZG9tKCkgKiBwb3NzaWJsZS5sZW5ndGgpKTtcclxuXHJcbiAgcmV0dXJuIHRleHQ7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=