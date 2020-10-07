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
    if (clean_deep__WEBPACK_IMPORTED_MODULE_4___default()(overlaidJSROOTPlot.fHists.arr).length === overlaidJSROOTPlot.fHists.arr.length //need fix: the first selected hist is not drewn at at all. Just when the second one is selected- the both of them are drewn.
    ) {
        drawJSROOT(histogramParam, id, overlaidJSROOTPlot);
        console.log('drew');
      }
  }, [updated_by_not_older_than, data, params_for_api.lumi, params_for_api.overlay_plot, params_for_api.dataset_name, params_for_api.run_number, params_for_api.normalize, selected_plot.name]);

  var _useBlinkOnUpdate = Object(_hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_12__["useBlinkOnUpdate"])(),
      blink = _useBlinkOnUpdate.blink;

  return __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_8__["StyledCol"], {
    space: 2,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 146,
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
      lineNumber: 147,
      columnNumber: 7
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_8__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_9__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 156,
      columnNumber: 9
    }
  }, selected_plot.displayedName), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_8__["Column"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 159,
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
        lineNumber: 163,
        columnNumber: 19
      }
    }),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 160,
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
      lineNumber: 166,
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
      lineNumber: 172,
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

  var _useBlinkOnUpdate = Object(_hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_10__["useBlinkOnUpdate"])(),
      blink = _useBlinkOnUpdate.blink;

  Object(react__WEBPACK_IMPORTED_MODULE_2__["useEffect"])(function () {
    if (!!document.getElementById("".concat(id))) {
      //@ts-ignore
      drawJSROOT("".concat(id), data);
    }
  }, [data, params_for_api.lumi, updated_by_not_older_than, selected_plot.name, blink]);
  return __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_6__["StyledCol"], {
    space: 2,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 70,
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
      lineNumber: 71,
      columnNumber: 7
    }
  }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_6__["PlotNameCol"], {
    error: Object(_plot_singlePlot_utils__WEBPACK_IMPORTED_MODULE_7__["get_plot_error"])(selected_plot).toString(),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 79,
      columnNumber: 9
    }
  }, selected_plot.displayedName), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_6__["Column"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 82,
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
        lineNumber: 86,
        columnNumber: 19
      }
    }),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 83,
      columnNumber: 11
    }
  })), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_6__["ImageDiv"], {
    id: "".concat(id),
    width: params_for_api.width,
    height: params_for_api.height,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 89,
      columnNumber: 9
    }
  })));
};

_s(ZoomedJSROOTPlot, "sbS3EC1SYLKmLzEjKQAheeptrpA=", false, function () {
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
  var the_lats_char = pathName.charAt(pathName.length - 1);

  if (the_lats_char !== '/') {
    pathName = pathName + '/';
  }

  console.log(pathName);
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy96b29tZWRPdmVybGF5UGxvdHMvem9vbWVkT3ZlcmxhaWRKU1JPT1RQbG90LnRzeCIsIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy96b29tZWRQbG90cy96b29tZWRKU1JPT1RQbG90LnRzeCIsIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy91dGlscy50cyJdLCJuYW1lcyI6WyJkcmF3SlNST09UIiwiaGlzdG9ncmFtUGFyYW0iLCJpZCIsIm92ZXJsYWlkSlNST09UUGxvdCIsIkpTUk9PVCIsImNsZWFudXAiLCJkcmF3IiwicGFyc2UiLCJKU09OIiwic3RyaW5naWZ5IiwiWm9vbWVkT3ZlcmxhaWRKU1JPT1RQbG90Iiwic2VsZWN0ZWRfcGxvdCIsInBhcmFtc19mb3JfYXBpIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJ1c2VSZXF1ZXN0IiwiZ2V0X2pyb290X3Bsb3QiLCJuYW1lIiwiZGF0YSIsIm92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzIiwib3ZlcmxheV9wbG90IiwibWFwIiwicGxvdCIsImNvcHkiLCJkYXRhc2V0X25hbWUiLCJydW5fbnVtYmVyIiwibHVtaSIsInB1c2giLCJsZW5ndGgiLCJoaXN0b2dyYW0xIiwiQ3JlYXRlVEhTdGFjayIsImhpc3RvZ3JhbTIiLCJoaXN0b2dyYW0zIiwiaGlzdG9ncmFtNCIsIlJlYWN0IiwidXNlQ29udGV4dCIsInN0b3JlIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsIm5vcm1hbGl6ZSIsInVzZUVmZmVjdCIsImNsZWFuRGVlcCIsImZIaXN0cyIsImFyciIsImNvbnNvbGUiLCJsb2ciLCJ1c2VCbGlua09uVXBkYXRlIiwiYmxpbmsiLCJ0b1N0cmluZyIsImZ1bmN0aW9uc19jb25maWciLCJtb2RlIiwiaGVpZ2h0Iiwid2lkdGgiLCJnZXRfcGxvdF9lcnJvciIsImRpc3BsYXllZE5hbWUiLCJyZW1vdmVQbG90RnJvbVJpZ2h0U2lkZSIsImRpc3BsYXkiLCJkb2N1bWVudCIsImdldEVsZW1lbnRCeUlkIiwiWm9vbWVkSlNST09UUGxvdCIsInNlcGVyYXRlUnVuQW5kTHVtaUluU2VhcmNoIiwicnVuQW5kTHVtaSIsInJ1bkFuZEx1bWlBcnJheSIsInNwbGl0IiwicGFyc2VkUnVuIiwicGFyc2VkTHVtaSIsInBhcnNlSW50IiwiZ2V0X2xhYmVsIiwiaW5mbyIsInZhbHVlIiwiZlN0cmluZyIsInR5cGUiLCJtaWxpc2VjIiwiRGF0ZSIsInRpbWUiLCJ0b1VUQ1N0cmluZyIsImdldFBhdGhOYW1lIiwiaXNCcm93c2VyIiwicGF0aE5hbWUiLCJ3aW5kb3ciLCJsb2NhdGlvbiIsInBhdGhuYW1lIiwidGhlX2xhdHNfY2hhciIsImNoYXJBdCIsIm1ha2VpZCIsInRleHQiLCJwb3NzaWJsZSIsImkiLCJNYXRoIiwiZmxvb3IiLCJyYW5kb20iXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFFQTtBQU9BO0FBQ0E7QUFDQTtBQVFBO0FBSUE7QUFDQTtBQUNBOztBQU9BLElBQU1BLFVBQVU7QUFBQSw4TEFBRyxpQkFDakJDLGNBRGlCLEVBRWpCQyxFQUZpQixFQUdqQkMsa0JBSGlCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLG1CQU1YQyxNQUFNLENBQUNDLE9BQVAsV0FBa0JKLGNBQWxCLFNBQW1DQyxFQUFuQyxFQU5XOztBQUFBO0FBT2pCO0FBQ0FFLGtCQUFNLENBQUNFLElBQVAsV0FDS0wsY0FETCxTQUNzQkMsRUFEdEIsR0FFRTtBQUNBRSxrQkFBTSxDQUFDRyxLQUFQLENBQWFDLElBQUksQ0FBQ0MsU0FBTCxDQUFlTixrQkFBZixDQUFiLENBSEYsWUFJS0YsY0FKTDs7QUFSaUI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsR0FBSDs7QUFBQSxrQkFBVkQsVUFB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QUNELENBUk0iLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNDc3ZTVjYzA4N2RhYWUzZWU5ODQuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCBjbGVhbkRlZXAgZnJvbSAnY2xlYW4tZGVlcCc7XHJcbmltcG9ydCB7IHVzZVJvdXRlciB9IGZyb20gJ25leHQvcm91dGVyJztcclxuXHJcbmltcG9ydCB7IGdldF9qcm9vdF9wbG90LCBmdW5jdGlvbnNfY29uZmlnIH0gZnJvbSAnLi4vLi4vLi4vLi4vY29uZmlnL2NvbmZpZyc7XHJcbmltcG9ydCB7XHJcbiAgUGFyYW1zRm9yQXBpUHJvcHMsXHJcbiAgVHJpcGxlUHJvcHMsXHJcbiAgUGxvdERhdGFQcm9wcyxcclxuICBRdWVyeVByb3BzLFxyXG59IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHsgdXNlUmVxdWVzdCB9IGZyb20gJy4uLy4uLy4uLy4uL2hvb2tzL3VzZVJlcXVlc3QnO1xyXG5pbXBvcnQgeyB1c2VFZmZlY3QgfSBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7XHJcbiAgU3R5bGVkQ29sLFxyXG4gIENvbHVtbixcclxuICBTdHlsZWRQbG90Um93LFxyXG4gIFBsb3ROYW1lQ29sLFxyXG4gIE1pbnVzSWNvbixcclxuICBJbWFnZURpdixcclxufSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7XHJcbiAgcmVtb3ZlUGxvdEZyb21SaWdodFNpZGUsXHJcbiAgZ2V0X3Bsb3RfZXJyb3IsXHJcbn0gZnJvbSAnLi4vLi4vcGxvdC9zaW5nbGVQbG90L3V0aWxzJztcclxuaW1wb3J0IHsgQnV0dG9uIH0gZnJvbSAnYW50ZCc7XHJcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0JztcclxuaW1wb3J0IHsgdXNlQmxpbmtPblVwZGF0ZSB9IGZyb20gJy4uLy4uLy4uLy4uL2hvb2tzL3VzZUJsaW5rT25VcGRhdGUnO1xyXG5cclxuaW50ZXJmYWNlIFpvb21lZEpTUk9PVFBsb3RzUHJvcHMge1xyXG4gIHNlbGVjdGVkX3Bsb3Q6IFBsb3REYXRhUHJvcHM7XHJcbiAgcGFyYW1zX2Zvcl9hcGk6IFBhcmFtc0ZvckFwaVByb3BzO1xyXG4gIGlkOiBzdHJpbmc7XHJcbn1cclxuY29uc3QgZHJhd0pTUk9PVCA9IGFzeW5jIChcclxuICBoaXN0b2dyYW1QYXJhbTogc3RyaW5nLFxyXG4gIGlkOiBzdHJpbmcsXHJcbiAgb3ZlcmxhaWRKU1JPT1RQbG90OiBhbnlcclxuKSA9PiB7XHJcbiAgLy9AdHMtaWdub3JlXHJcbiAgYXdhaXQgSlNST09ULmNsZWFudXAoYCR7aGlzdG9ncmFtUGFyYW19JHtpZH1gKTtcclxuICAvL0B0cy1pZ25vcmVcclxuICBKU1JPT1QuZHJhdyhcclxuICAgIGAke2hpc3RvZ3JhbVBhcmFtfSR7aWR9YCxcclxuICAgIC8vQHRzLWlnbm9yZVxyXG4gICAgSlNST09ULnBhcnNlKEpTT04uc3RyaW5naWZ5KG92ZXJsYWlkSlNST09UUGxvdCkpLFxyXG4gICAgYCR7aGlzdG9ncmFtUGFyYW19YFxyXG4gICk7XHJcbn07XHJcblxyXG5leHBvcnQgY29uc3QgWm9vbWVkT3ZlcmxhaWRKU1JPT1RQbG90ID0gKHtcclxuICBzZWxlY3RlZF9wbG90LFxyXG4gIHBhcmFtc19mb3JfYXBpLFxyXG4gIGlkLFxyXG59OiBab29tZWRKU1JPT1RQbG90c1Byb3BzKSA9PiB7XHJcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XHJcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XHJcblxyXG4gIGNvbnN0IHsgZGF0YSB9ID0gdXNlUmVxdWVzdChnZXRfanJvb3RfcGxvdChwYXJhbXNfZm9yX2FwaSksIHt9LCBbXHJcbiAgICBzZWxlY3RlZF9wbG90Lm5hbWUsXHJcbiAgXSk7XHJcblxyXG4gIGNvbnN0IG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzOiBhbnlbXSA9IHBhcmFtc19mb3JfYXBpPy5vdmVybGF5X3Bsb3RcclxuICAgID8gcGFyYW1zX2Zvcl9hcGkub3ZlcmxheV9wbG90Lm1hcCgocGxvdDogVHJpcGxlUHJvcHMpID0+IHtcclxuICAgICAgICBjb25zdCBjb3B5OiBhbnkgPSB7IC4uLnBhcmFtc19mb3JfYXBpIH07XHJcblxyXG4gICAgICAgIGlmIChwbG90LmRhdGFzZXRfbmFtZSkge1xyXG4gICAgICAgICAgY29weS5kYXRhc2V0X25hbWUgPSBwbG90LmRhdGFzZXRfbmFtZTtcclxuICAgICAgICB9XHJcbiAgICAgICAgY29weS5ydW5fbnVtYmVyID0gcGxvdC5ydW5fbnVtYmVyO1xyXG4gICAgICAgIGNvbnN0IHsgZGF0YSB9ID0gdXNlUmVxdWVzdChnZXRfanJvb3RfcGxvdChjb3B5KSwge30sIFtcclxuICAgICAgICAgIHNlbGVjdGVkX3Bsb3QubmFtZSxcclxuICAgICAgICAgIHF1ZXJ5Lmx1bWksXHJcbiAgICAgICAgXSk7XHJcbiAgICAgICAgcmV0dXJuIGRhdGE7XHJcbiAgICAgIH0pXHJcbiAgICA6IFtdO1xyXG5cclxuICBvdmVybGFpZF9wbG90c19ydW5zX2FuZF9kYXRhc2V0cy5wdXNoKGRhdGEpO1xyXG5cclxuICBsZXQgb3ZlcmxhaWRKU1JPT1RQbG90OiBhbnkgPSB7fTtcclxuXHJcbiAgLy9jaGVja2luZyBob3cgbWFueSBoaXN0b2dyYW1zIGFyZSBvdmVybGFpZCwgYmVjYXVzZSBqdXN0IHNlcGFyYXRlZCBvYmplY3RzXHJcbiAgLy8gKGkuZSBzZXBhcmF0ZSB2YXJpYWJsZXMgKSB0byBKU1JPT1QuQ3JlYXRlVEhTdGFjaygpIGZ1bmNcclxuICBpZiAob3ZlcmxhaWRfcGxvdHNfcnVuc19hbmRfZGF0YXNldHMubGVuZ3RoID09PSAwKSB7XHJcbiAgICByZXR1cm4gbnVsbDtcclxuICB9IGVsc2UgaWYgKG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzLmxlbmd0aCA9PT0gMSkge1xyXG4gICAgY29uc3QgaGlzdG9ncmFtMSA9IG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzWzBdO1xyXG4gICAgLy9AdHMtaWdub3JlXHJcbiAgICBvdmVybGFpZEpTUk9PVFBsb3QgPSBKU1JPT1QuQ3JlYXRlVEhTdGFjayhoaXN0b2dyYW0xKTtcclxuICB9IGVsc2UgaWYgKG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzLmxlbmd0aCA9PT0gMikge1xyXG4gICAgY29uc3QgaGlzdG9ncmFtMSA9IG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzWzBdO1xyXG4gICAgY29uc3QgaGlzdG9ncmFtMiA9IG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzWzFdO1xyXG4gICAgLy9AdHMtaWdub3JlXHJcbiAgICBvdmVybGFpZEpTUk9PVFBsb3QgPSBKU1JPT1QuQ3JlYXRlVEhTdGFjayhoaXN0b2dyYW0xLCBoaXN0b2dyYW0yKTtcclxuICB9IGVsc2UgaWYgKG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzLmxlbmd0aCA9PT0gMykge1xyXG4gICAgY29uc3QgaGlzdG9ncmFtMSA9IG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzWzBdO1xyXG4gICAgY29uc3QgaGlzdG9ncmFtMiA9IG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzWzFdO1xyXG4gICAgY29uc3QgaGlzdG9ncmFtMyA9IG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzWzJdO1xyXG4gICAgLy9AdHMtaWdub3JlXHJcbiAgICBvdmVybGFpZEpTUk9PVFBsb3QgPSBKU1JPT1QuQ3JlYXRlVEhTdGFjayhcclxuICAgICAgaGlzdG9ncmFtMSxcclxuICAgICAgaGlzdG9ncmFtMixcclxuICAgICAgaGlzdG9ncmFtM1xyXG4gICAgKTtcclxuICB9IGVsc2UgaWYgKG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzLmxlbmd0aCA9PT0gNCkge1xyXG4gICAgY29uc3QgaGlzdG9ncmFtMSA9IG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzWzBdO1xyXG4gICAgY29uc3QgaGlzdG9ncmFtMiA9IG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzWzFdO1xyXG4gICAgY29uc3QgaGlzdG9ncmFtMyA9IG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzWzJdO1xyXG4gICAgY29uc3QgaGlzdG9ncmFtNCA9IG92ZXJsYWlkX3Bsb3RzX3J1bnNfYW5kX2RhdGFzZXRzWzNdO1xyXG4gICAgLy9AdHMtaWdub3JlXHJcbiAgICBvdmVybGFpZEpTUk9PVFBsb3QgPSBKU1JPT1QuQ3JlYXRlVEhTdGFjayhcclxuICAgICAgaGlzdG9ncmFtMSxcclxuICAgICAgaGlzdG9ncmFtMixcclxuICAgICAgaGlzdG9ncmFtMyxcclxuICAgICAgaGlzdG9ncmFtNFxyXG4gICAgKTtcclxuICB9XHJcbiAgY29uc3QgeyB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuIH0gPSBSZWFjdC51c2VDb250ZXh0KHN0b3JlKTtcclxuXHJcbiAgY29uc3QgaGlzdG9ncmFtUGFyYW0gPSBwYXJhbXNfZm9yX2FwaS5ub3JtYWxpemUgPyAnaGlzdCcgOiAnbm9zdGFjayc7XHJcbiAgLy9tYWtlIHN1cmUgdGhhdCBubyBudWxsIGhpc3RvZ3JhbXMgYXJlIHBhc3NlZCB0byBkcmF3IGZ1bmMuXHJcbiAgLy9vbiBmaXJzdCwgc2Vjb25kIHJlbmVkZXIgb3ZlcmxhaWRKU1JPT1RQbG90LmZIaXN0cy5hcnIgaXMgW251bGwsIG51bGxdXHJcbiAgLy9AdHMtaWdub3JlXHJcbiAgdXNlRWZmZWN0KCgpID0+IHtcclxuICAgIGlmIChcclxuICAgICAgY2xlYW5EZWVwKG92ZXJsYWlkSlNST09UUGxvdC5mSGlzdHMuYXJyKS5sZW5ndGggPT09XHJcbiAgICAgIG92ZXJsYWlkSlNST09UUGxvdC5mSGlzdHMuYXJyLmxlbmd0aCAvL25lZWQgZml4OiB0aGUgZmlyc3Qgc2VsZWN0ZWQgaGlzdCBpcyBub3QgZHJld24gYXQgYXQgYWxsLiBKdXN0IHdoZW4gdGhlIHNlY29uZCBvbmUgaXMgc2VsZWN0ZWQtIHRoZSBib3RoIG9mIHRoZW0gYXJlIGRyZXduLlxyXG4gICAgKSB7XHJcbiAgICAgIGRyYXdKU1JPT1QoaGlzdG9ncmFtUGFyYW0sIGlkLCBvdmVybGFpZEpTUk9PVFBsb3QpO1xyXG4gICAgICBjb25zb2xlLmxvZygnZHJldycpXHJcblxyXG4gICAgfVxyXG4gIH0sIFtcclxuICAgIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4sXHJcbiAgICBkYXRhLFxyXG4gICAgcGFyYW1zX2Zvcl9hcGkubHVtaSxcclxuICAgIHBhcmFtc19mb3JfYXBpLm92ZXJsYXlfcGxvdCxcclxuICAgIHBhcmFtc19mb3JfYXBpLmRhdGFzZXRfbmFtZSxcclxuICAgIHBhcmFtc19mb3JfYXBpLnJ1bl9udW1iZXIsXHJcbiAgICBwYXJhbXNfZm9yX2FwaS5ub3JtYWxpemUsXHJcbiAgICBzZWxlY3RlZF9wbG90Lm5hbWVcclxuICBdKTtcclxuICBjb25zdCB7IGJsaW5rIH0gPSB1c2VCbGlua09uVXBkYXRlKCk7XHJcbiAgcmV0dXJuIChcclxuICAgIDxTdHlsZWRDb2wgc3BhY2U9ezJ9PlxyXG4gICAgICA8U3R5bGVkUGxvdFJvd1xyXG4gICAgICAgIGlzTG9hZGluZz17YmxpbmsudG9TdHJpbmcoKX1cclxuICAgICAgICBhbmltYXRpb249eyhmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnKS50b1N0cmluZygpfVxyXG4gICAgICAgIG1pbmhlaWdodD17cGFyYW1zX2Zvcl9hcGkuaGVpZ2h0fVxyXG4gICAgICAgIHdpZHRoPXtwYXJhbXNfZm9yX2FwaS53aWR0aD8udG9TdHJpbmcoKX1cclxuICAgICAgICBpc19wbG90X3NlbGVjdGVkPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgbm9wb2ludGVyPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgICAganVzdGlmeWNvbnRlbnQ9XCJjZW50ZXJcIlxyXG4gICAgICA+XHJcbiAgICAgICAgPFBsb3ROYW1lQ29sIGVycm9yPXtnZXRfcGxvdF9lcnJvcihzZWxlY3RlZF9wbG90KS50b1N0cmluZygpfT5cclxuICAgICAgICAgIHtzZWxlY3RlZF9wbG90LmRpc3BsYXllZE5hbWV9XHJcbiAgICAgICAgPC9QbG90TmFtZUNvbD5cclxuICAgICAgICA8Q29sdW1uPlxyXG4gICAgICAgICAgPEJ1dHRvblxyXG4gICAgICAgICAgICB0eXBlPVwibGlua1wiXHJcbiAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHJlbW92ZVBsb3RGcm9tUmlnaHRTaWRlKHF1ZXJ5LCBzZWxlY3RlZF9wbG90KX1cclxuICAgICAgICAgICAgaWNvbj17PE1pbnVzSWNvbiAvPn1cclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgPC9Db2x1bW4+XHJcbiAgICAgICAgPEltYWdlRGl2XHJcbiAgICAgICAgICBzdHlsZT17eyBkaXNwbGF5OiBwYXJhbXNfZm9yX2FwaS5ub3JtYWxpemUgPyAnJyA6ICdub25lJyB9fVxyXG4gICAgICAgICAgaWQ9e2BoaXN0JHtpZH1gfVxyXG4gICAgICAgICAgd2lkdGg9e3BhcmFtc19mb3JfYXBpLndpZHRofVxyXG4gICAgICAgICAgaGVpZ2h0PXtwYXJhbXNfZm9yX2FwaS5oZWlnaHR9XHJcbiAgICAgICAgLz5cclxuICAgICAgICA8SW1hZ2VEaXZcclxuICAgICAgICAgIHN0eWxlPXt7IGRpc3BsYXk6IHBhcmFtc19mb3JfYXBpLm5vcm1hbGl6ZSA/ICdub25lJyA6ICcnIH19XHJcbiAgICAgICAgICBpZD17YG5vc3RhY2ske2lkfWB9XHJcbiAgICAgICAgICB3aWR0aD17cGFyYW1zX2Zvcl9hcGkud2lkdGh9XHJcbiAgICAgICAgICBoZWlnaHQ9e3BhcmFtc19mb3JfYXBpLmhlaWdodH1cclxuICAgICAgICAvPlxyXG4gICAgICA8L1N0eWxlZFBsb3RSb3c+XHJcbiAgICA8L1N0eWxlZENvbD5cclxuICApO1xyXG59O1xyXG4iLCJpbXBvcnQgUmVhY3QsIHsgdXNlRWZmZWN0IH0gZnJvbSAncmVhY3QnO1xyXG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XHJcblxyXG5pbXBvcnQgeyBnZXRfanJvb3RfcGxvdCwgZnVuY3Rpb25zX2NvbmZpZyB9IGZyb20gJy4uLy4uLy4uLy4uL2NvbmZpZy9jb25maWcnO1xyXG5pbXBvcnQge1xyXG4gIFBhcmFtc0ZvckFwaVByb3BzLFxyXG4gIFBsb3REYXRhUHJvcHMsXHJcbiAgUXVlcnlQcm9wcyxcclxufSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IHVzZVJlcXVlc3QgfSBmcm9tICcuLi8uLi8uLi8uLi9ob29rcy91c2VSZXF1ZXN0JztcclxuaW1wb3J0IHtcclxuICBTdHlsZWRDb2wsXHJcbiAgU3R5bGVkUGxvdFJvdyxcclxuICBQbG90TmFtZUNvbCxcclxuICBNaW51c0ljb24sXHJcbiAgQ29sdW1uLFxyXG4gIEltYWdlRGl2LFxyXG59IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHtcclxuICByZW1vdmVQbG90RnJvbVJpZ2h0U2lkZSxcclxuICBnZXRfcGxvdF9lcnJvcixcclxufSBmcm9tICcuLi8uLi9wbG90L3NpbmdsZVBsb3QvdXRpbHMnO1xyXG5pbXBvcnQgeyBCdXR0b24gfSBmcm9tICdhbnRkJztcclxuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi8uLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnO1xyXG5pbXBvcnQgeyB1c2VCbGlua09uVXBkYXRlIH0gZnJvbSAnLi4vLi4vLi4vLi4vaG9va3MvdXNlQmxpbmtPblVwZGF0ZSc7XHJcblxyXG5pbnRlcmZhY2UgWm9vbWVkSlNST09UUGxvdHNQcm9wcyB7XHJcbiAgc2VsZWN0ZWRfcGxvdDogUGxvdERhdGFQcm9wcztcclxuICBwYXJhbXNfZm9yX2FwaTogUGFyYW1zRm9yQXBpUHJvcHM7XHJcbiAgaWQ6IHN0cmluZztcclxufVxyXG5cclxuY29uc3QgZHJhd0pTUk9PVCA9IGFzeW5jIChpZDogc3RyaW5nLCBkYXRhOiBhbnkpID0+IHtcclxuICAvL2luIG9yZGVyIHRvIGdldCBuZXcgSlNST09UIHBsb3QsIGZpcnN0IG9mIGFsbCB3ZSBuZWVkIHRvIGNsZWFuIGRpdiB3aXRoIG9sZCBwbG90XHJcbiAgaWYgKCEhZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpKSB7XHJcbiAgICAvL0B0cy1pZ25vcmVcclxuICAgIGF3YWl0IEpTUk9PVC5jbGVhbnVwKGlkKTtcclxuICAgIC8vYWZ0ZXIgY2xlYW51cCB3ZSBjYW4gZHJhdyBhIG5ldyBwbG90XHJcbiAgICAvL0B0cy1pZ25vcmVcclxuICAgIEpTUk9PVC5kcmF3KGlkLCBKU1JPT1QucGFyc2UoSlNPTi5zdHJpbmdpZnkoZGF0YSkpLCAnaGlzdCcpO1xyXG4gIH1cclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBab29tZWRKU1JPT1RQbG90ID0gKHtcclxuICBzZWxlY3RlZF9wbG90LFxyXG4gIHBhcmFtc19mb3JfYXBpLFxyXG4gIGlkLFxyXG59OiBab29tZWRKU1JPT1RQbG90c1Byb3BzKSA9PiB7XHJcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XHJcbiAgY29uc3QgcXVlcnk6IFF1ZXJ5UHJvcHMgPSByb3V0ZXIucXVlcnk7XHJcbiAgLy8gY29uc3QgaWQgPSBtYWtlaWQoKVxyXG4gIGNvbnN0IHsgZGF0YSB9ID0gdXNlUmVxdWVzdChnZXRfanJvb3RfcGxvdChwYXJhbXNfZm9yX2FwaSksIHt9LCBbXHJcbiAgICBzZWxlY3RlZF9wbG90Lm5hbWUsXHJcbiAgICBwYXJhbXNfZm9yX2FwaS5sdW1pLFxyXG4gIF0pO1xyXG5cclxuICBjb25zdCB7IHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4gfSA9IFJlYWN0LnVzZUNvbnRleHQoc3RvcmUpO1xyXG5cclxuICBjb25zdCB7IGJsaW5rIH0gPSB1c2VCbGlua09uVXBkYXRlKCk7XHJcblxyXG4gIHVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBpZiAoISFkb2N1bWVudC5nZXRFbGVtZW50QnlJZChgJHtpZH1gKSkge1xyXG5cclxuICAgICAgLy9AdHMtaWdub3JlXHJcbiAgICAgIGRyYXdKU1JPT1QoYCR7aWR9YCwgZGF0YSk7XHJcbiAgICB9XHJcbiAgfSwgW2RhdGEsIHBhcmFtc19mb3JfYXBpLmx1bWksIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4sIHNlbGVjdGVkX3Bsb3QubmFtZSwgYmxpbmtdKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxTdHlsZWRDb2wgc3BhY2U9ezJ9PlxyXG4gICAgICA8U3R5bGVkUGxvdFJvd1xyXG4gICAgICAgIGlzTG9hZGluZz17YmxpbmsudG9TdHJpbmcoKX1cclxuICAgICAgICBhbmltYXRpb249eyhmdW5jdGlvbnNfY29uZmlnLm1vZGUgPT09ICdPTkxJTkUnKS50b1N0cmluZygpfVxyXG4gICAgICAgIG1pbmhlaWdodD17cGFyYW1zX2Zvcl9hcGkuaGVpZ2h0fVxyXG4gICAgICAgIHdpZHRoPXtwYXJhbXNfZm9yX2FwaS53aWR0aD8udG9TdHJpbmcoKX1cclxuICAgICAgICBpc19wbG90X3NlbGVjdGVkPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgbm9wb2ludGVyPXt0cnVlLnRvU3RyaW5nKCl9XHJcbiAgICAgID5cclxuICAgICAgICA8UGxvdE5hbWVDb2wgZXJyb3I9e2dldF9wbG90X2Vycm9yKHNlbGVjdGVkX3Bsb3QpLnRvU3RyaW5nKCl9PlxyXG4gICAgICAgICAge3NlbGVjdGVkX3Bsb3QuZGlzcGxheWVkTmFtZX1cclxuICAgICAgICA8L1Bsb3ROYW1lQ29sPlxyXG4gICAgICAgIDxDb2x1bW4+XHJcbiAgICAgICAgICA8QnV0dG9uXHJcbiAgICAgICAgICAgIHR5cGU9XCJsaW5rXCJcclxuICAgICAgICAgICAgb25DbGljaz17KCkgPT4gcmVtb3ZlUGxvdEZyb21SaWdodFNpZGUocXVlcnksIHNlbGVjdGVkX3Bsb3QpfVxyXG4gICAgICAgICAgICBpY29uPXs8TWludXNJY29uIC8+fVxyXG4gICAgICAgICAgLz5cclxuICAgICAgICA8L0NvbHVtbj5cclxuICAgICAgICA8SW1hZ2VEaXZcclxuICAgICAgICAgIGlkPXtgJHtpZH1gfVxyXG4gICAgICAgICAgd2lkdGg9e3BhcmFtc19mb3JfYXBpLndpZHRofVxyXG4gICAgICAgICAgaGVpZ2h0PXtwYXJhbXNfZm9yX2FwaS5oZWlnaHR9XHJcbiAgICAgICAgLz5cclxuICAgICAgPC9TdHlsZWRQbG90Um93PlxyXG4gICAgPC9TdHlsZWRDb2w+XHJcbiAgKTtcclxufTtcclxuIiwiaW1wb3J0IHsgSW5mb1Byb3BzIH0gZnJvbSAnLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5cclxuZXhwb3J0IGNvbnN0IHNlcGVyYXRlUnVuQW5kTHVtaUluU2VhcmNoID0gKHJ1bkFuZEx1bWk6IHN0cmluZykgPT4ge1xyXG4gIGNvbnN0IHJ1bkFuZEx1bWlBcnJheSA9IHJ1bkFuZEx1bWkuc3BsaXQoJzonKTtcclxuICBjb25zdCBwYXJzZWRSdW4gPSBydW5BbmRMdW1pQXJyYXlbMF07XHJcbiAgY29uc3QgcGFyc2VkTHVtaSA9IHJ1bkFuZEx1bWlBcnJheVsxXSA/IHBhcnNlSW50KHJ1bkFuZEx1bWlBcnJheVsxXSkgOiAwO1xyXG5cclxuICByZXR1cm4geyBwYXJzZWRSdW4sIHBhcnNlZEx1bWkgfTtcclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRfbGFiZWwgPSAoaW5mbzogSW5mb1Byb3BzLCBkYXRhPzogYW55KSA9PiB7XHJcbiAgY29uc3QgdmFsdWUgPSBkYXRhID8gZGF0YS5mU3RyaW5nIDogbnVsbDtcclxuXHJcbiAgaWYgKGluZm8/LnR5cGUgJiYgaW5mby50eXBlID09PSAndGltZScgJiYgdmFsdWUpIHtcclxuICAgIGNvbnN0IG1pbGlzZWMgPSBuZXcgRGF0ZShwYXJzZUludCh2YWx1ZSkgKiAxMDAwKTtcclxuICAgIGNvbnN0IHRpbWUgPSBtaWxpc2VjLnRvVVRDU3RyaW5nKCk7XHJcbiAgICByZXR1cm4gdGltZTtcclxuICB9IGVsc2Uge1xyXG4gICAgcmV0dXJuIHZhbHVlID8gdmFsdWUgOiAnTm8gaW5mb3JtYXRpb24nO1xyXG4gIH1cclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBnZXRQYXRoTmFtZSA9ICgpID0+IHtcclxuICBjb25zdCBpc0Jyb3dzZXIgPSAoKSA9PiB0eXBlb2Ygd2luZG93ICE9PSAndW5kZWZpbmVkJztcclxuICBsZXQgcGF0aE5hbWUgPSAoaXNCcm93c2VyKCkgJiYgd2luZG93LmxvY2F0aW9uLnBhdGhuYW1lKSB8fCAnLyc7XHJcbiAgY29uc3QgdGhlX2xhdHNfY2hhciA9IHBhdGhOYW1lLmNoYXJBdChwYXRoTmFtZS5sZW5ndGgtMSk7XHJcbiAgaWYodGhlX2xhdHNfY2hhciAhPT0gJy8nKXtcclxuICAgIHBhdGhOYW1lID0gcGF0aE5hbWUgKyAnLydcclxuICB9XHJcbiAgY29uc29sZS5sb2cocGF0aE5hbWUpXHJcbiAgcmV0dXJuIHBhdGhOYW1lO1xyXG59O1xyXG5leHBvcnQgY29uc3QgbWFrZWlkID0gKCkgPT4ge1xyXG4gIHZhciB0ZXh0ID0gJyc7XHJcbiAgdmFyIHBvc3NpYmxlID0gJ0FCQ0RFRkdISUpLTE1OT1BRUlNUVVZXWFlaYWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXonO1xyXG5cclxuICBmb3IgKHZhciBpID0gMDsgaSA8IDU7IGkrKylcclxuICAgIHRleHQgKz0gcG9zc2libGUuY2hhckF0KE1hdGguZmxvb3IoTWF0aC5yYW5kb20oKSAqIHBvc3NpYmxlLmxlbmd0aCkpO1xyXG5cclxuICByZXR1cm4gdGV4dDtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==