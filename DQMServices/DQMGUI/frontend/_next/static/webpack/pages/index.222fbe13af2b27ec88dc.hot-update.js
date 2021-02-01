webpackHotUpdate_N_E("pages/index",{

/***/ "./components/plots/plot/plotsWithLayouts/oneLayout.tsx":
/*!**************************************************************!*\
  !*** ./components/plots/plot/plotsWithLayouts/oneLayout.tsx ***!
  \**************************************************************/
/*! exports provided: OnePlotInLayout */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "OnePlotInLayout", function() { return OnePlotInLayout; });
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../../../../config/config */ "./config/config.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../../../hooks/useBlinkOnUpdate */ "./hooks/useBlinkOnUpdate.tsx");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./styledComponents */ "./components/plots/plot/plotsWithLayouts/styledComponents.ts");
/* harmony import */ var _plot__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./plot */ "./components/plots/plot/plotsWithLayouts/plot.tsx");


var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/plots/plot/plotsWithLayouts/oneLayout.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_1__["createElement"];






var OnePlotInLayout = function OnePlotInLayout(_ref) {
  _s();

  var plots = _ref.plots,
      globalState = _ref.globalState,
      imageRefScrollDown = _ref.imageRefScrollDown,
      layoutName = _ref.layoutName,
      query = _ref.query,
      selected_plots = _ref.selected_plots;

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_1__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_3__["store"]),
      size = _React$useContext.size;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_1__["useState"](layoutName),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState, 2),
      nameOfLayout = _React$useState2[0],
      setNameOfLayout = _React$useState2[1];

  var imageRef = react__WEBPACK_IMPORTED_MODULE_1__["useRef"](null);
  react__WEBPACK_IMPORTED_MODULE_1__["useEffect"](function () {
    setNameOfLayout(layoutName);
  }, [layoutName]);
  var plotsAmount = //in order to get tidy layout, has to be x^2 plots in one layout. In the layuts, where the plot number is 
  //less than x^2, we're adding peseudo plots (empty divs)
  Math.ceil(Math.log(2) / Math.log(plots.length)) - Math.log(2) / Math.log(plots.length) !== 0 && plots.length !== 1 // log(2)/log(1)=0, that's we need to avoid to add pseudo plots in layout when is just 1 plot in it
  //exception: need to plots.length^2, because when there is 2 plots in layout, we want to display it like 4 (2 real in 2 pseudo plots)
  // otherwise it won't fit in parent div.
  ? plots.length + Math.ceil(Math.sqrt(plots.length)) : Math.pow(plots.length, 2);
  var layoutArea = size.h * size.w;
  var ratio = size.w / size.h;
  var onePlotArea = layoutArea / plotsAmount;
  var onePlotHeight = Math.floor(Math.sqrt(onePlotArea / ratio));
  var onePlotWidth = Math.floor(Math.sqrt(onePlotArea / ratio) * ratio);
  var howMuchInOneLine = Math.floor(size.w / onePlotWidth);
  var auto = [];
  var i;

  var _useBlinkOnUpdate = Object(_hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_4__["useBlinkOnUpdate"])(),
      blink = _useBlinkOnUpdate.blink,
      updated_by_not_older_than = _useBlinkOnUpdate.updated_by_not_older_than;

  for (i = 0; i < howMuchInOneLine; i++) {
    auto.push('auto');
  }

  return __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_5__["ParentWrapper"], {
    isLoading: blink.toString(),
    animation: (_config_config__WEBPACK_IMPORTED_MODULE_2__["functions_config"].mode === 'ONLINE').toString(),
    size: size,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 50,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_5__["LayoutName"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 54,
      columnNumber: 7
    }
  }, decodeURI(nameOfLayout)), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_5__["LayoutWrapper"], {
    size: size,
    auto: auto.join(' '),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 55,
      columnNumber: 7
    }
  }, plots.map(function (plot) {
    return __jsx(_plot__WEBPACK_IMPORTED_MODULE_6__["Plot"], {
      globalState: globalState,
      query: query,
      plot: plot,
      onePlotHeight: onePlotHeight,
      onePlotWidth: onePlotWidth,
      selected_plots: selected_plots,
      imageRef: imageRef,
      imageRefScrollDown: imageRefScrollDown,
      blink: blink,
      updated_by_not_older_than: updated_by_not_older_than,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 62,
        columnNumber: 15
      }
    });
  })));
};

_s(OnePlotInLayout, "R/l4RVDjvvg/XNbyxoqZySTV5H8=", false, function () {
  return [_hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_4__["useBlinkOnUpdate"]];
});

_c = OnePlotInLayout;

var _c;

$RefreshReg$(_c, "OnePlotInLayout");

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

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy9wbG90L3Bsb3RzV2l0aExheW91dHMvb25lTGF5b3V0LnRzeCJdLCJuYW1lcyI6WyJPbmVQbG90SW5MYXlvdXQiLCJwbG90cyIsImdsb2JhbFN0YXRlIiwiaW1hZ2VSZWZTY3JvbGxEb3duIiwibGF5b3V0TmFtZSIsInF1ZXJ5Iiwic2VsZWN0ZWRfcGxvdHMiLCJSZWFjdCIsInN0b3JlIiwic2l6ZSIsIm5hbWVPZkxheW91dCIsInNldE5hbWVPZkxheW91dCIsImltYWdlUmVmIiwicGxvdHNBbW91bnQiLCJNYXRoIiwiY2VpbCIsImxvZyIsImxlbmd0aCIsInNxcnQiLCJsYXlvdXRBcmVhIiwiaCIsInciLCJyYXRpbyIsIm9uZVBsb3RBcmVhIiwib25lUGxvdEhlaWdodCIsImZsb29yIiwib25lUGxvdFdpZHRoIiwiaG93TXVjaEluT25lTGluZSIsImF1dG8iLCJpIiwidXNlQmxpbmtPblVwZGF0ZSIsImJsaW5rIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsInB1c2giLCJ0b1N0cmluZyIsImZ1bmN0aW9uc19jb25maWciLCJtb2RlIiwiZGVjb2RlVVJJIiwiam9pbiIsIm1hcCIsInBsb3QiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQVdPLElBQU1BLGVBQWUsR0FBRyxTQUFsQkEsZUFBa0IsT0FBb0c7QUFBQTs7QUFBQSxNQUFqR0MsS0FBaUcsUUFBakdBLEtBQWlHO0FBQUEsTUFBMUZDLFdBQTBGLFFBQTFGQSxXQUEwRjtBQUFBLE1BQTdFQyxrQkFBNkUsUUFBN0VBLGtCQUE2RTtBQUFBLE1BQXpEQyxVQUF5RCxRQUF6REEsVUFBeUQ7QUFBQSxNQUE3Q0MsS0FBNkMsUUFBN0NBLEtBQTZDO0FBQUEsTUFBdENDLGNBQXNDLFFBQXRDQSxjQUFzQzs7QUFBQSwwQkFDaEhDLGdEQUFBLENBQWlCQywrREFB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LFVBQUNDLElBQUQsRUFBVTtBQUNsQixXQUNFLE1BQUMsMENBQUQ7QUFDRSxpQkFBVyxFQUFFdEMsV0FEZjtBQUVFLFdBQUssRUFBRUcsS0FGVDtBQUdFLFVBQUksRUFBRW1DLElBSFI7QUFJRSxtQkFBYSxFQUFFaEIsYUFKakI7QUFLRSxrQkFBWSxFQUFFRSxZQUxoQjtBQU1FLG9CQUFjLEVBQUVwQixjQU5sQjtBQU9FLGNBQVEsRUFBRU0sUUFQWjtBQVFFLHdCQUFrQixFQUFFVCxrQkFSdEI7QUFTRSxXQUFLLEVBQUU0QixLQVRUO0FBVUUsK0JBQXlCLEVBQUVDLHlCQVY3QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BREY7QUFhRCxHQWRELENBTEosQ0FMRixDQURGO0FBNkJELENBNURNOztHQUFNaEMsZTtVQXlCa0M4Qix3RTs7O0tBekJsQzlCLGUiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguMjIyZmJlMTNhZjJiMjdlYzg4ZGMuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0J1xyXG5cclxuaW1wb3J0IHsgZnVuY3Rpb25zX2NvbmZpZyB9IGZyb20gJy4uLy4uLy4uLy4uL2NvbmZpZy9jb25maWcnXHJcbmltcG9ydCB7IHN0b3JlIH0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGV4dHMvbGVmdFNpZGVDb250ZXh0J1xyXG5pbXBvcnQgeyB1c2VCbGlua09uVXBkYXRlIH0gZnJvbSAnLi4vLi4vLi4vLi4vaG9va3MvdXNlQmxpbmtPblVwZGF0ZSdcclxuaW1wb3J0IHsgTGF5b3V0TmFtZSwgTGF5b3V0V3JhcHBlciwgUGFyZW50V3JhcHBlciB9IGZyb20gJy4vc3R5bGVkQ29tcG9uZW50cydcclxuaW1wb3J0IHsgUGxvdCB9IGZyb20gJy4vcGxvdCdcclxuXHJcbmludGVyZmFjZSBPbmVQbG90SW5MYXlvdXQge1xyXG4gIGxheW91dE5hbWU6IHN0cmluZztcclxuICBwbG90czogYW55W107XHJcbiAgc2VsZWN0ZWRfcGxvdHM6IGFueSxcclxuICBnbG9iYWxTdGF0ZTogYW55LFxyXG4gIGltYWdlUmVmU2Nyb2xsRG93bjogYW55LFxyXG4gIHF1ZXJ5OiBhbnksXHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBPbmVQbG90SW5MYXlvdXQgPSAoeyBwbG90cywgZ2xvYmFsU3RhdGUsIGltYWdlUmVmU2Nyb2xsRG93biwgbGF5b3V0TmFtZSwgcXVlcnksIHNlbGVjdGVkX3Bsb3RzIH06IE9uZVBsb3RJbkxheW91dCkgPT4ge1xyXG4gIGNvbnN0IHsgc2l6ZSB9ID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSlcclxuICBjb25zdCBbbmFtZU9mTGF5b3V0LCBzZXROYW1lT2ZMYXlvdXRdID0gUmVhY3QudXNlU3RhdGUobGF5b3V0TmFtZSlcclxuICBjb25zdCBpbWFnZVJlZiA9IFJlYWN0LnVzZVJlZihudWxsKTtcclxuXHJcbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcclxuICAgIHNldE5hbWVPZkxheW91dChsYXlvdXROYW1lKVxyXG4gIH0sIFtsYXlvdXROYW1lXSlcclxuICBjb25zdCBwbG90c0Ftb3VudCA9XHJcbiAgICAvL2luIG9yZGVyIHRvIGdldCB0aWR5IGxheW91dCwgaGFzIHRvIGJlIHheMiBwbG90cyBpbiBvbmUgbGF5b3V0LiBJbiB0aGUgbGF5dXRzLCB3aGVyZSB0aGUgcGxvdCBudW1iZXIgaXMgXHJcbiAgICAvL2xlc3MgdGhhbiB4XjIsIHdlJ3JlIGFkZGluZyBwZXNldWRvIHBsb3RzIChlbXB0eSBkaXZzKVxyXG4gICAgKE1hdGguY2VpbChNYXRoLmxvZygyKSAvIE1hdGgubG9nKHBsb3RzLmxlbmd0aCkpIC0gKE1hdGgubG9nKDIpIC8gTWF0aC5sb2cocGxvdHMubGVuZ3RoKSkgIT09IDApICYmXHJcbiAgICAgIHBsb3RzLmxlbmd0aCAhPT0gMSAvLyBsb2coMikvbG9nKDEpPTAsIHRoYXQncyB3ZSBuZWVkIHRvIGF2b2lkIHRvIGFkZCBwc2V1ZG8gcGxvdHMgaW4gbGF5b3V0IHdoZW4gaXMganVzdCAxIHBsb3QgaW4gaXRcclxuICAgICAgLy9leGNlcHRpb246IG5lZWQgdG8gcGxvdHMubGVuZ3RoXjIsIGJlY2F1c2Ugd2hlbiB0aGVyZSBpcyAyIHBsb3RzIGluIGxheW91dCwgd2Ugd2FudCB0byBkaXNwbGF5IGl0IGxpa2UgNCAoMiByZWFsIGluIDIgcHNldWRvIHBsb3RzKVxyXG4gICAgICAvLyBvdGhlcndpc2UgaXQgd29uJ3QgZml0IGluIHBhcmVudCBkaXYuXHJcbiAgICAgID8gcGxvdHMubGVuZ3RoICsgTWF0aC5jZWlsKE1hdGguc3FydChwbG90cy5sZW5ndGgpKSA6IHBsb3RzLmxlbmd0aCAqKiAyXHJcblxyXG4gIGNvbnN0IGxheW91dEFyZWEgPSBzaXplLmggKiBzaXplLndcclxuICBjb25zdCByYXRpbyA9IHNpemUudyAvIHNpemUuaFxyXG4gIGNvbnN0IG9uZVBsb3RBcmVhID0gbGF5b3V0QXJlYSAvIHBsb3RzQW1vdW50XHJcbiAgY29uc3Qgb25lUGxvdEhlaWdodCA9IE1hdGguZmxvb3IoTWF0aC5zcXJ0KG9uZVBsb3RBcmVhIC8gcmF0aW8pKVxyXG4gIGNvbnN0IG9uZVBsb3RXaWR0aCA9IE1hdGguZmxvb3IoTWF0aC5zcXJ0KG9uZVBsb3RBcmVhIC8gcmF0aW8pICogcmF0aW8pXHJcbiAgY29uc3QgaG93TXVjaEluT25lTGluZSA9IE1hdGguZmxvb3Ioc2l6ZS53IC8gb25lUGxvdFdpZHRoKVxyXG4gIGNvbnN0IGF1dG8gPSBbXVxyXG4gIHZhciBpO1xyXG4gIGNvbnN0IHsgYmxpbmssIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4gfSA9IHVzZUJsaW5rT25VcGRhdGUoKTtcclxuXHJcbiAgZm9yIChpID0gMDsgaSA8IGhvd011Y2hJbk9uZUxpbmU7IGkrKykge1xyXG4gICAgYXV0by5wdXNoKCdhdXRvJylcclxuICB9XHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8UGFyZW50V3JhcHBlclxyXG4gICAgICBpc0xvYWRpbmc9e2JsaW5rLnRvU3RyaW5nKCl9XHJcbiAgICAgIGFuaW1hdGlvbj17KGZ1bmN0aW9uc19jb25maWcubW9kZSA9PT0gJ09OTElORScpLnRvU3RyaW5nKCl9XHJcbiAgICAgIHNpemU9e3NpemV9PlxyXG4gICAgICA8TGF5b3V0TmFtZT57ZGVjb2RlVVJJKG5hbWVPZkxheW91dCl9PC9MYXlvdXROYW1lPlxyXG4gICAgICA8TGF5b3V0V3JhcHBlclxyXG4gICAgICAgIHNpemU9e3NpemV9XHJcbiAgICAgICAgYXV0bz17YXV0by5qb2luKCcgJyl9XHJcbiAgICAgID5cclxuICAgICAgICB7XHJcbiAgICAgICAgICBwbG90cy5tYXAoKHBsb3QpID0+IHtcclxuICAgICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgICA8UGxvdFxyXG4gICAgICAgICAgICAgICAgZ2xvYmFsU3RhdGU9e2dsb2JhbFN0YXRlfVxyXG4gICAgICAgICAgICAgICAgcXVlcnk9e3F1ZXJ5fVxyXG4gICAgICAgICAgICAgICAgcGxvdD17cGxvdH1cclxuICAgICAgICAgICAgICAgIG9uZVBsb3RIZWlnaHQ9e29uZVBsb3RIZWlnaHR9XHJcbiAgICAgICAgICAgICAgICBvbmVQbG90V2lkdGg9e29uZVBsb3RXaWR0aH1cclxuICAgICAgICAgICAgICAgIHNlbGVjdGVkX3Bsb3RzPXtzZWxlY3RlZF9wbG90c31cclxuICAgICAgICAgICAgICAgIGltYWdlUmVmPXtpbWFnZVJlZn1cclxuICAgICAgICAgICAgICAgIGltYWdlUmVmU2Nyb2xsRG93bj17aW1hZ2VSZWZTY3JvbGxEb3dufVxyXG4gICAgICAgICAgICAgICAgYmxpbms9e2JsaW5rfVxyXG4gICAgICAgICAgICAgICAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbj17dXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbn0gLz5cclxuICAgICAgICAgICAgKVxyXG4gICAgICAgICAgfSl9XHJcbiAgICAgIDwvTGF5b3V0V3JhcHBlcj5cclxuICAgIDwvUGFyZW50V3JhcHBlcj5cclxuICApXHJcbn0gIl0sInNvdXJjZVJvb3QiOiIifQ==