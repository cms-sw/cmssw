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
    plotsAmount: plots.length,
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
      lineNumber: 55,
      columnNumber: 7
    }
  }, decodeURI(nameOfLayout)), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_5__["LayoutWrapper"], {
    size: size,
    auto: auto.join(' '),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 56,
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
        lineNumber: 63,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy9wbG90L3Bsb3RzV2l0aExheW91dHMvb25lTGF5b3V0LnRzeCJdLCJuYW1lcyI6WyJPbmVQbG90SW5MYXlvdXQiLCJwbG90cyIsImdsb2JhbFN0YXRlIiwiaW1hZ2VSZWZTY3JvbGxEb3duIiwibGF5b3V0TmFtZSIsInF1ZXJ5Iiwic2VsZWN0ZWRfcGxvdHMiLCJSZWFjdCIsInN0b3JlIiwic2l6ZSIsIm5hbWVPZkxheW91dCIsInNldE5hbWVPZkxheW91dCIsImltYWdlUmVmIiwicGxvdHNBbW91bnQiLCJNYXRoIiwiY2VpbCIsImxvZyIsImxlbmd0aCIsInNxcnQiLCJsYXlvdXRBcmVhIiwiaCIsInciLCJyYXRpbyIsIm9uZVBsb3RBcmVhIiwib25lUGxvdEhlaWdodCIsImZsb29yIiwib25lUGxvdFdpZHRoIiwiaG93TXVjaEluT25lTGluZSIsImF1dG8iLCJpIiwidXNlQmxpbmtPblVwZGF0ZSIsImJsaW5rIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsInB1c2giLCJ0b1N0cmluZyIsImZ1bmN0aW9uc19jb25maWciLCJtb2RlIiwiZGVjb2RlVVJJIiwiam9pbiIsIm1hcCIsInBsb3QiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQVdPLElBQU1BLGVBQWUsR0FBRyxTQUFsQkEsZUFBa0IsT0FBb0c7QUFBQTs7QUFBQSxNQUFqR0MsS0FBaUcsUUFBakdBLEtBQWlHO0FBQUEsTUFBMUZDLFdBQTBGLFFBQTFGQSxXQUEwRjtBQUFBLE1BQTdFQyxrQkFBNkUsUUFBN0VBLGtCQUE2RTtBQUFBLE1BQXpEQyxVQUF5RCxRQUF6REEsVUFBeUQ7QUFBQSxNQUE3Q0MsS0FBNkMsUUFBN0NBLEtBQTZDO0FBQUEsTUFBdENDLGNBQXNDLFFBQXRDQSxjQUFzQzs7QUFBQSwwQkFDaEhDLGdEQUFBLENBQWlCQywrREFBakIsQ0FEZ0g7QUFBQSxNQUN6SEMsSUFEeUgscUJBQ3pIQSxJQUR5SDs7QUFBQSx3QkFFekZGLDhDQUFBLENBQWVILFVBQWYsQ0FGeUY7QUFBQTtBQUFBLE1BRTFITSxZQUYwSDtBQUFBLE1BRTVHQyxlQUY0Rzs7QUFHakksTUFBTUMsUUFBUSxHQUFHTCw0Q0FBQSxDQUFhLElBQWIsQ0FBakI7QUFFQUEsaURBQUEsQ0FBZ0IsWUFBTTtBQUNwQkksbUJBQWUsQ0FBQ1AsVUFBRCxDQUFmO0FBQ0QsR0FGRCxFQUVHLENBQUNBLFVBQUQsQ0FGSDtBQUdBLE1BQU1TLFdBQVcsR0FDZjtBQUNBO0FBQ0NDLE1BQUksQ0FBQ0MsSUFBTCxDQUFVRCxJQUFJLENBQUNFLEdBQUwsQ0FBUyxDQUFULElBQWNGLElBQUksQ0FBQ0UsR0FBTCxDQUFTZixLQUFLLENBQUNnQixNQUFmLENBQXhCLElBQW1ESCxJQUFJLENBQUNFLEdBQUwsQ0FBUyxDQUFULElBQWNGLElBQUksQ0FBQ0UsR0FBTCxDQUFTZixLQUFLLENBQUNnQixNQUFmLENBQWpFLEtBQTZGLENBQTlGLElBQ0VoQixLQUFLLENBQUNnQixNQUFOLEtBQWlCLENBRG5CLENBQ3FCO0FBQ25CO0FBQ0E7QUFIRixJQUlJaEIsS0FBSyxDQUFDZ0IsTUFBTixHQUFlSCxJQUFJLENBQUNDLElBQUwsQ0FBVUQsSUFBSSxDQUFDSSxJQUFMLENBQVVqQixLQUFLLENBQUNnQixNQUFoQixDQUFWLENBSm5CLFlBSXdEaEIsS0FBSyxDQUFDZ0IsTUFKOUQsRUFJd0UsQ0FKeEUsQ0FIRjtBQVNBLE1BQU1FLFVBQVUsR0FBR1YsSUFBSSxDQUFDVyxDQUFMLEdBQVNYLElBQUksQ0FBQ1ksQ0FBakM7QUFDQSxNQUFNQyxLQUFLLEdBQUdiLElBQUksQ0FBQ1ksQ0FBTCxHQUFTWixJQUFJLENBQUNXLENBQTVCO0FBQ0EsTUFBTUcsV0FBVyxHQUFHSixVQUFVLEdBQUdOLFdBQWpDO0FBQ0EsTUFBTVcsYUFBYSxHQUFHVixJQUFJLENBQUNXLEtBQUwsQ0FBV1gsSUFBSSxDQUFDSSxJQUFMLENBQVVLLFdBQVcsR0FBR0QsS0FBeEIsQ0FBWCxDQUF0QjtBQUNBLE1BQU1JLFlBQVksR0FBR1osSUFBSSxDQUFDVyxLQUFMLENBQVdYLElBQUksQ0FBQ0ksSUFBTCxDQUFVSyxXQUFXLEdBQUdELEtBQXhCLElBQWlDQSxLQUE1QyxDQUFyQjtBQUNBLE1BQU1LLGdCQUFnQixHQUFHYixJQUFJLENBQUNXLEtBQUwsQ0FBV2hCLElBQUksQ0FBQ1ksQ0FBTCxHQUFTSyxZQUFwQixDQUF6QjtBQUNBLE1BQU1FLElBQUksR0FBRyxFQUFiO0FBQ0EsTUFBSUMsQ0FBSjs7QUF4QmlJLDBCQXlCcEZDLGdGQUFnQixFQXpCb0U7QUFBQSxNQXlCekhDLEtBekJ5SCxxQkF5QnpIQSxLQXpCeUg7QUFBQSxNQXlCbEhDLHlCQXpCa0gscUJBeUJsSEEseUJBekJrSDs7QUEyQmpJLE9BQUtILENBQUMsR0FBRyxDQUFULEVBQVlBLENBQUMsR0FBR0YsZ0JBQWhCLEVBQWtDRSxDQUFDLEVBQW5DLEVBQXVDO0FBQ3JDRCxRQUFJLENBQUNLLElBQUwsQ0FBVSxNQUFWO0FBQ0Q7O0FBRUQsU0FDRSxNQUFDLCtEQUFEO0FBQ0UsZUFBVyxFQUFFaEMsS0FBSyxDQUFDZ0IsTUFEckI7QUFFRSxhQUFTLEVBQUVjLEtBQUssQ0FBQ0csUUFBTixFQUZiO0FBR0UsYUFBUyxFQUFFLENBQUNDLCtEQUFnQixDQUFDQyxJQUFqQixLQUEwQixRQUEzQixFQUFxQ0YsUUFBckMsRUFIYjtBQUlFLFFBQUksRUFBRXpCLElBSlI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUtFLE1BQUMsNERBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUFhNEIsU0FBUyxDQUFDM0IsWUFBRCxDQUF0QixDQUxGLEVBTUUsTUFBQywrREFBRDtBQUNFLFFBQUksRUFBRUQsSUFEUjtBQUVFLFFBQUksRUFBRW1CLElBQUksQ0FBQ1UsSUFBTCxDQUFVLEdBQVYsQ0FGUjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBS0lyQyxLQUFLLENBQUNzQyxHQUFOLENBQVUsVUFBQ0MsSUFBRCxFQUFVO0FBQ2xCLFdBQ0UsTUFBQywwQ0FBRDtBQUNFLGlCQUFXLEVBQUV0QyxXQURmO0FBRUUsV0FBSyxFQUFFRyxLQUZUO0FBR0UsVUFBSSxFQUFFbUMsSUFIUjtBQUlFLG1CQUFhLEVBQUVoQixhQUpqQjtBQUtFLGtCQUFZLEVBQUVFLFlBTGhCO0FBTUUsb0JBQWMsRUFBRXBCLGNBTmxCO0FBT0UsY0FBUSxFQUFFTSxRQVBaO0FBUUUsd0JBQWtCLEVBQUVULGtCQVJ0QjtBQVNFLFdBQUssRUFBRTRCLEtBVFQ7QUFVRSwrQkFBeUIsRUFBRUMseUJBVjdCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFERjtBQWFELEdBZEQsQ0FMSixDQU5GLENBREY7QUE4QkQsQ0E3RE07O0dBQU1oQyxlO1VBeUJrQzhCLHdFOzs7S0F6QmxDOUIsZSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4zODJhZjQxMTExZDkxOTcyMmJlMC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnXHJcblxyXG5pbXBvcnQgeyBmdW5jdGlvbnNfY29uZmlnIH0gZnJvbSAnLi4vLi4vLi4vLi4vY29uZmlnL2NvbmZpZydcclxuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi8uLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnXHJcbmltcG9ydCB7IHVzZUJsaW5rT25VcGRhdGUgfSBmcm9tICcuLi8uLi8uLi8uLi9ob29rcy91c2VCbGlua09uVXBkYXRlJ1xyXG5pbXBvcnQgeyBMYXlvdXROYW1lLCBMYXlvdXRXcmFwcGVyLCBQYXJlbnRXcmFwcGVyIH0gZnJvbSAnLi9zdHlsZWRDb21wb25lbnRzJ1xyXG5pbXBvcnQgeyBQbG90IH0gZnJvbSAnLi9wbG90J1xyXG5cclxuaW50ZXJmYWNlIE9uZVBsb3RJbkxheW91dCB7XHJcbiAgbGF5b3V0TmFtZTogc3RyaW5nO1xyXG4gIHBsb3RzOiBhbnlbXTtcclxuICBzZWxlY3RlZF9wbG90czogYW55LFxyXG4gIGdsb2JhbFN0YXRlOiBhbnksXHJcbiAgaW1hZ2VSZWZTY3JvbGxEb3duOiBhbnksXHJcbiAgcXVlcnk6IGFueSxcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IE9uZVBsb3RJbkxheW91dCA9ICh7IHBsb3RzLCBnbG9iYWxTdGF0ZSwgaW1hZ2VSZWZTY3JvbGxEb3duLCBsYXlvdXROYW1lLCBxdWVyeSwgc2VsZWN0ZWRfcGxvdHMgfTogT25lUGxvdEluTGF5b3V0KSA9PiB7XHJcbiAgY29uc3QgeyBzaXplIH0gPSBSZWFjdC51c2VDb250ZXh0KHN0b3JlKVxyXG4gIGNvbnN0IFtuYW1lT2ZMYXlvdXQsIHNldE5hbWVPZkxheW91dF0gPSBSZWFjdC51c2VTdGF0ZShsYXlvdXROYW1lKVxyXG4gIGNvbnN0IGltYWdlUmVmID0gUmVhY3QudXNlUmVmKG51bGwpO1xyXG5cclxuICBSZWFjdC51c2VFZmZlY3QoKCkgPT4ge1xyXG4gICAgc2V0TmFtZU9mTGF5b3V0KGxheW91dE5hbWUpXHJcbiAgfSwgW2xheW91dE5hbWVdKVxyXG4gIGNvbnN0IHBsb3RzQW1vdW50ID1cclxuICAgIC8vaW4gb3JkZXIgdG8gZ2V0IHRpZHkgbGF5b3V0LCBoYXMgdG8gYmUgeF4yIHBsb3RzIGluIG9uZSBsYXlvdXQuIEluIHRoZSBsYXl1dHMsIHdoZXJlIHRoZSBwbG90IG51bWJlciBpcyBcclxuICAgIC8vbGVzcyB0aGFuIHheMiwgd2UncmUgYWRkaW5nIHBlc2V1ZG8gcGxvdHMgKGVtcHR5IGRpdnMpXHJcbiAgICAoTWF0aC5jZWlsKE1hdGgubG9nKDIpIC8gTWF0aC5sb2cocGxvdHMubGVuZ3RoKSkgLSAoTWF0aC5sb2coMikgLyBNYXRoLmxvZyhwbG90cy5sZW5ndGgpKSAhPT0gMCkgJiZcclxuICAgICAgcGxvdHMubGVuZ3RoICE9PSAxIC8vIGxvZygyKS9sb2coMSk9MCwgdGhhdCdzIHdlIG5lZWQgdG8gYXZvaWQgdG8gYWRkIHBzZXVkbyBwbG90cyBpbiBsYXlvdXQgd2hlbiBpcyBqdXN0IDEgcGxvdCBpbiBpdFxyXG4gICAgICAvL2V4Y2VwdGlvbjogbmVlZCB0byBwbG90cy5sZW5ndGheMiwgYmVjYXVzZSB3aGVuIHRoZXJlIGlzIDIgcGxvdHMgaW4gbGF5b3V0LCB3ZSB3YW50IHRvIGRpc3BsYXkgaXQgbGlrZSA0ICgyIHJlYWwgaW4gMiBwc2V1ZG8gcGxvdHMpXHJcbiAgICAgIC8vIG90aGVyd2lzZSBpdCB3b24ndCBmaXQgaW4gcGFyZW50IGRpdi5cclxuICAgICAgPyBwbG90cy5sZW5ndGggKyBNYXRoLmNlaWwoTWF0aC5zcXJ0KHBsb3RzLmxlbmd0aCkpIDogcGxvdHMubGVuZ3RoICoqIDJcclxuXHJcbiAgY29uc3QgbGF5b3V0QXJlYSA9IHNpemUuaCAqIHNpemUud1xyXG4gIGNvbnN0IHJhdGlvID0gc2l6ZS53IC8gc2l6ZS5oXHJcbiAgY29uc3Qgb25lUGxvdEFyZWEgPSBsYXlvdXRBcmVhIC8gcGxvdHNBbW91bnRcclxuICBjb25zdCBvbmVQbG90SGVpZ2h0ID0gTWF0aC5mbG9vcihNYXRoLnNxcnQob25lUGxvdEFyZWEgLyByYXRpbykpXHJcbiAgY29uc3Qgb25lUGxvdFdpZHRoID0gTWF0aC5mbG9vcihNYXRoLnNxcnQob25lUGxvdEFyZWEgLyByYXRpbykgKiByYXRpbylcclxuICBjb25zdCBob3dNdWNoSW5PbmVMaW5lID0gTWF0aC5mbG9vcihzaXplLncgLyBvbmVQbG90V2lkdGgpXHJcbiAgY29uc3QgYXV0byA9IFtdXHJcbiAgdmFyIGk7XHJcbiAgY29uc3QgeyBibGluaywgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiB9ID0gdXNlQmxpbmtPblVwZGF0ZSgpO1xyXG5cclxuICBmb3IgKGkgPSAwOyBpIDwgaG93TXVjaEluT25lTGluZTsgaSsrKSB7XHJcbiAgICBhdXRvLnB1c2goJ2F1dG8nKVxyXG4gIH1cclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxQYXJlbnRXcmFwcGVyXHJcbiAgICAgIHBsb3RzQW1vdW50PXtwbG90cy5sZW5ndGh9XHJcbiAgICAgIGlzTG9hZGluZz17YmxpbmsudG9TdHJpbmcoKX1cclxuICAgICAgYW5pbWF0aW9uPXsoZnVuY3Rpb25zX2NvbmZpZy5tb2RlID09PSAnT05MSU5FJykudG9TdHJpbmcoKX1cclxuICAgICAgc2l6ZT17c2l6ZX0+XHJcbiAgICAgIDxMYXlvdXROYW1lPntkZWNvZGVVUkkobmFtZU9mTGF5b3V0KX08L0xheW91dE5hbWU+XHJcbiAgICAgIDxMYXlvdXRXcmFwcGVyXHJcbiAgICAgICAgc2l6ZT17c2l6ZX1cclxuICAgICAgICBhdXRvPXthdXRvLmpvaW4oJyAnKX1cclxuICAgICAgPlxyXG4gICAgICAgIHtcclxuICAgICAgICAgIHBsb3RzLm1hcCgocGxvdCkgPT4ge1xyXG4gICAgICAgICAgICByZXR1cm4gKFxyXG4gICAgICAgICAgICAgIDxQbG90XHJcbiAgICAgICAgICAgICAgICBnbG9iYWxTdGF0ZT17Z2xvYmFsU3RhdGV9XHJcbiAgICAgICAgICAgICAgICBxdWVyeT17cXVlcnl9XHJcbiAgICAgICAgICAgICAgICBwbG90PXtwbG90fVxyXG4gICAgICAgICAgICAgICAgb25lUGxvdEhlaWdodD17b25lUGxvdEhlaWdodH1cclxuICAgICAgICAgICAgICAgIG9uZVBsb3RXaWR0aD17b25lUGxvdFdpZHRofVxyXG4gICAgICAgICAgICAgICAgc2VsZWN0ZWRfcGxvdHM9e3NlbGVjdGVkX3Bsb3RzfVxyXG4gICAgICAgICAgICAgICAgaW1hZ2VSZWY9e2ltYWdlUmVmfVxyXG4gICAgICAgICAgICAgICAgaW1hZ2VSZWZTY3JvbGxEb3duPXtpbWFnZVJlZlNjcm9sbERvd259XHJcbiAgICAgICAgICAgICAgICBibGluaz17Ymxpbmt9XHJcbiAgICAgICAgICAgICAgICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuPXt1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFufSAvPlxyXG4gICAgICAgICAgICApXHJcbiAgICAgICAgICB9KX1cclxuICAgICAgPC9MYXlvdXRXcmFwcGVyPlxyXG4gICAgPC9QYXJlbnRXcmFwcGVyPlxyXG4gIClcclxufSAiXSwic291cmNlUm9vdCI6IiJ9