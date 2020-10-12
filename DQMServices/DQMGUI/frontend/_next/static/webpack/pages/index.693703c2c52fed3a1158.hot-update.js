webpackHotUpdate_N_E("pages/index",{

/***/ "./components/plots/plot/plotImage.tsx":
/*!*********************************************!*\
  !*** ./components/plots/plot/plotImage.tsx ***!
  \*********************************************/
/*! exports provided: PlotImage */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "PlotImage", function() { return PlotImage; });
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/regenerator */ "./node_modules/@babel/runtime/regenerator/index.js");
/* harmony import */ var _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/asyncToGenerator */ "./node_modules/@babel/runtime/helpers/esm/asyncToGenerator.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../../config/config */ "./config/config.ts");
/* harmony import */ var _errorMessage__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../errorMessage */ "./components/plots/errorMessage.tsx");
/* harmony import */ var _imageFallback__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../imageFallback */ "./components/plots/imageFallback.tsx");
/* harmony import */ var _singlePlot_utils__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ./singlePlot/utils */ "./components/plots/plot/singlePlot/utils.ts");




var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/plots/plot/plotImage.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_3__["createElement"];





var PlotImage = function PlotImage(_ref) {
  _s();

  var imageRef = _ref.imageRef,
      query = _ref.query,
      isPlotSelected = _ref.isPlotSelected,
      params_for_api = _ref.params_for_api,
      plotURL = _ref.plotURL,
      updated_by_not_older_than = _ref.updated_by_not_older_than,
      blink = _ref.blink,
      plot = _ref.plot;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_3__["useState"]("".concat(_config_config__WEBPACK_IMPORTED_MODULE_4__["root_url"]).concat(plotURL, ";notOlderThan=").concat(updated_by_not_older_than)),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__["default"])(_React$useState, 2),
      new_image_url = _React$useState2[0],
      set_new_image_url = _React$useState2[1];

  var _React$useState3 = react__WEBPACK_IMPORTED_MODULE_3__["useState"]("".concat(_config_config__WEBPACK_IMPORTED_MODULE_4__["root_url"]).concat(plotURL, ";notOlderThan=").concat(updated_by_not_older_than)),
      _React$useState4 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__["default"])(_React$useState3, 2),
      old_image_url = _React$useState4[0],
      set_old_image_url = _React$useState4[1];

  var _React$useState5 = react__WEBPACK_IMPORTED_MODULE_3__["useState"](true),
      _React$useState6 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__["default"])(_React$useState5, 2),
      show_old_img = _React$useState6[0],
      set_show_old_img = _React$useState6[1];

  var _React$useState7 = react__WEBPACK_IMPORTED_MODULE_3__["useState"](false),
      _React$useState8 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_2__["default"])(_React$useState7, 2),
      imageError = _React$useState8[0],
      setImageError = _React$useState8[1];

  react__WEBPACK_IMPORTED_MODULE_3__["useEffect"](function () {
    set_new_image_url("".concat(_config_config__WEBPACK_IMPORTED_MODULE_4__["root_url"]).concat(plotURL, ";notOlderThan=").concat(updated_by_not_older_than));
    set_show_old_img(blink);
  }, [updated_by_not_older_than, params_for_api.customizeProps, params_for_api.height, params_for_api.width, params_for_api.run_number, params_for_api.dataset_name, params_for_api.lumi, params_for_api.normalize]);
  var old_image_display = show_old_img ? '' : 'none';
  var new_image_display = show_old_img ? 'none' : '';
  return __jsx(react__WEBPACK_IMPORTED_MODULE_3__["Fragment"], null, imageError ? __jsx(_errorMessage__WEBPACK_IMPORTED_MODULE_5__["ErrorMessage"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 68,
      columnNumber: 9
    }
  }) : __jsx("div", {
    onClick: /*#__PURE__*/Object(_babel_runtime_helpers_esm_asyncToGenerator__WEBPACK_IMPORTED_MODULE_1__["default"])( /*#__PURE__*/_babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.mark(function _callee() {
      return _babel_runtime_regenerator__WEBPACK_IMPORTED_MODULE_0___default.a.wrap(function _callee$(_context) {
        while (1) {
          switch (_context.prev = _context.next) {
            case 0:
              if (!imageRef) {
                _context.next = 9;
                break;
              }

              if (!isPlotSelected) {
                _context.next = 6;
                break;
              }

              _context.next = 4;
              return Object(_singlePlot_utils__WEBPACK_IMPORTED_MODULE_7__["removePlotFromRightSide"])(query, plot);

            case 4:
              _context.next = 8;
              break;

            case 6:
              _context.next = 8;
              return Object(_singlePlot_utils__WEBPACK_IMPORTED_MODULE_7__["addPlotToRightSide"])(query, plot);

            case 8:
              scroll(imageRef);

            case 9:
            case "end":
              return _context.stop();
          }
        }
      }, _callee);
    })),
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 70,
      columnNumber: 9
    }
  }, !imageError && __jsx(react__WEBPACK_IMPORTED_MODULE_3__["Fragment"], null, __jsx(_imageFallback__WEBPACK_IMPORTED_MODULE_6__["ImageFallback"], {
    retryTimes: 3,
    style: {
      display: new_image_display
    },
    onLoad: function onLoad() {
      set_old_image_url(new_image_url);
      set_show_old_img(false);
    },
    alt: plot.name,
    src: new_image_url,
    setImageError: setImageError,
    width: params_for_api.width,
    height: params_for_api.height,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 82,
      columnNumber: 15
    }
  }), __jsx(_imageFallback__WEBPACK_IMPORTED_MODULE_6__["ImageFallback"], {
    retryTimes: 3,
    style: {
      display: old_image_display
    },
    alt: plot.name,
    src: old_image_url,
    setImageError: setImageError,
    width: 'auto',
    height: 'auto',
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 99,
      columnNumber: 15
    }
  }))));
};

_s(PlotImage, "02JUSGUXbPaqnAriS4/tfOckOvI=");

_c = PlotImage;

var _c;

$RefreshReg$(_c, "PlotImage");

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

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy9wbG90L3Bsb3RJbWFnZS50c3giXSwibmFtZXMiOlsiUGxvdEltYWdlIiwiaW1hZ2VSZWYiLCJxdWVyeSIsImlzUGxvdFNlbGVjdGVkIiwicGFyYW1zX2Zvcl9hcGkiLCJwbG90VVJMIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsImJsaW5rIiwicGxvdCIsIlJlYWN0Iiwicm9vdF91cmwiLCJuZXdfaW1hZ2VfdXJsIiwic2V0X25ld19pbWFnZV91cmwiLCJvbGRfaW1hZ2VfdXJsIiwic2V0X29sZF9pbWFnZV91cmwiLCJzaG93X29sZF9pbWciLCJzZXRfc2hvd19vbGRfaW1nIiwiaW1hZ2VFcnJvciIsInNldEltYWdlRXJyb3IiLCJjdXN0b21pemVQcm9wcyIsImhlaWdodCIsIndpZHRoIiwicnVuX251bWJlciIsImRhdGFzZXRfbmFtZSIsImx1bWkiLCJub3JtYWxpemUiLCJvbGRfaW1hZ2VfZGlzcGxheSIsIm5ld19pbWFnZV9kaXNwbGF5IiwicmVtb3ZlUGxvdEZyb21SaWdodFNpZGUiLCJhZGRQbG90VG9SaWdodFNpZGUiLCJzY3JvbGwiLCJkaXNwbGF5IiwibmFtZSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBRUE7QUFLQTtBQUNBO0FBQ0E7QUFnQk8sSUFBTUEsU0FBUyxHQUFHLFNBQVpBLFNBQVksT0FTSDtBQUFBOztBQUFBLE1BUnBCQyxRQVFvQixRQVJwQkEsUUFRb0I7QUFBQSxNQVBwQkMsS0FPb0IsUUFQcEJBLEtBT29CO0FBQUEsTUFOcEJDLGNBTW9CLFFBTnBCQSxjQU1vQjtBQUFBLE1BTHBCQyxjQUtvQixRQUxwQkEsY0FLb0I7QUFBQSxNQUpwQkMsT0FJb0IsUUFKcEJBLE9BSW9CO0FBQUEsTUFIcEJDLHlCQUdvQixRQUhwQkEseUJBR29CO0FBQUEsTUFGcEJDLEtBRW9CLFFBRnBCQSxLQUVvQjtBQUFBLE1BRHBCQyxJ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QUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBbEJGLENBWEosQ0FKSixDQURGO0FBaURELENBeEZNOztHQUFNbEIsUzs7S0FBQUEsUyIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC42OTM3MDNjMmM1MmZlZDNhMTE1OC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xuXG5pbXBvcnQgeyByb290X3VybCB9IGZyb20gJy4uLy4uLy4uL2NvbmZpZy9jb25maWcnO1xuaW1wb3J0IHtcbiAgUGFyYW1zRm9yQXBpUHJvcHMsXG4gIFF1ZXJ5UHJvcHMsXG59IGZyb20gJy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcbmltcG9ydCB7IEVycm9yTWVzc2FnZSB9IGZyb20gJy4uL2Vycm9yTWVzc2FnZSc7XG5pbXBvcnQgeyBJbWFnZUZhbGxiYWNrIH0gZnJvbSAnLi4vaW1hZ2VGYWxsYmFjayc7XG5pbXBvcnQge1xuICBhZGRQbG90VG9SaWdodFNpZGUsXG4gIHJlbW92ZVBsb3RGcm9tUmlnaHRTaWRlLFxufSBmcm9tICcuL3NpbmdsZVBsb3QvdXRpbHMnO1xuXG5pbnRlcmZhY2UgUGxvdEltYWdlUHJvcHMge1xuICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuOiBudW1iZXI7XG4gIHBhcmFtc19mb3JfYXBpOiBQYXJhbXNGb3JBcGlQcm9wcztcbiAgYmxpbms6IGJvb2xlYW47XG4gIHBsb3Q6IGFueTtcbiAgcGxvdFVSTDogc3RyaW5nO1xuICBpc1Bsb3RTZWxlY3RlZD86IGJvb2xlYW47XG4gIHF1ZXJ5OiBRdWVyeVByb3BzO1xuICBpbWFnZVJlZj86IGFueTtcbn1cblxuZXhwb3J0IGNvbnN0IFBsb3RJbWFnZSA9ICh7XG4gIGltYWdlUmVmLFxuICBxdWVyeSxcbiAgaXNQbG90U2VsZWN0ZWQsXG4gIHBhcmFtc19mb3JfYXBpLFxuICBwbG90VVJMLFxuICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuLFxuICBibGluayxcbiAgcGxvdCxcbn06IFBsb3RJbWFnZVByb3BzKSA9PiB7XG4gIGNvbnN0IFtuZXdfaW1hZ2VfdXJsLCBzZXRfbmV3X2ltYWdlX3VybF0gPSBSZWFjdC51c2VTdGF0ZShcbiAgICBgJHtyb290X3VybH0ke3Bsb3RVUkx9O25vdE9sZGVyVGhhbj0ke3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW59YFxuICApO1xuICBjb25zdCBbb2xkX2ltYWdlX3VybCwgc2V0X29sZF9pbWFnZV91cmxdID0gUmVhY3QudXNlU3RhdGUoXG4gICAgYCR7cm9vdF91cmx9JHtwbG90VVJMfTtub3RPbGRlclRoYW49JHt1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFufWBcbiAgKTtcblxuICBjb25zdCBbc2hvd19vbGRfaW1nLCBzZXRfc2hvd19vbGRfaW1nXSA9IFJlYWN0LnVzZVN0YXRlKHRydWUpO1xuICBjb25zdCBbaW1hZ2VFcnJvciwgc2V0SW1hZ2VFcnJvcl0gPSBSZWFjdC51c2VTdGF0ZShmYWxzZSk7XG5cbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcbiAgICBzZXRfbmV3X2ltYWdlX3VybChcbiAgICAgIGAke3Jvb3RfdXJsfSR7cGxvdFVSTH07bm90T2xkZXJUaGFuPSR7dXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbn1gXG4gICAgKTtcbiAgICBzZXRfc2hvd19vbGRfaW1nKGJsaW5rKTtcbiAgfSwgW1xuICAgIHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4sXG4gICAgcGFyYW1zX2Zvcl9hcGkuY3VzdG9taXplUHJvcHMsXG4gICAgcGFyYW1zX2Zvcl9hcGkuaGVpZ2h0LFxuICAgIHBhcmFtc19mb3JfYXBpLndpZHRoLFxuICAgIHBhcmFtc19mb3JfYXBpLnJ1bl9udW1iZXIsXG4gICAgcGFyYW1zX2Zvcl9hcGkuZGF0YXNldF9uYW1lLFxuICAgIHBhcmFtc19mb3JfYXBpLmx1bWksXG4gICAgcGFyYW1zX2Zvcl9hcGkubm9ybWFsaXplLFxuICBdKTtcblxuICBjb25zdCBvbGRfaW1hZ2VfZGlzcGxheSA9IHNob3dfb2xkX2ltZyA/ICcnIDogJ25vbmUnO1xuICBjb25zdCBuZXdfaW1hZ2VfZGlzcGxheSA9IHNob3dfb2xkX2ltZyA/ICdub25lJyA6ICcnO1xuXG4gIHJldHVybiAoXG4gICAgPD5cbiAgICAgIHtpbWFnZUVycm9yID8gKFxuICAgICAgICA8RXJyb3JNZXNzYWdlIC8+XG4gICAgICApIDogKFxuICAgICAgICA8ZGl2XG4gICAgICAgICAgb25DbGljaz17YXN5bmMgKCkgPT4ge1xuICAgICAgICAgICAgaWYgKGltYWdlUmVmKSB7XG4gICAgICAgICAgICAgIGlzUGxvdFNlbGVjdGVkXG4gICAgICAgICAgICAgICAgPyBhd2FpdCByZW1vdmVQbG90RnJvbVJpZ2h0U2lkZShxdWVyeSwgcGxvdClcbiAgICAgICAgICAgICAgICA6IGF3YWl0IGFkZFBsb3RUb1JpZ2h0U2lkZShxdWVyeSwgcGxvdCk7XG4gICAgICAgICAgICAgIHNjcm9sbChpbWFnZVJlZik7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfX1cbiAgICAgICAgPlxuICAgICAgICAgIHshaW1hZ2VFcnJvciAmJiAoXG4gICAgICAgICAgICA8PlxuICAgICAgICAgICAgICA8SW1hZ2VGYWxsYmFja1xuICAgICAgICAgICAgICAgIHJldHJ5VGltZXM9ezN9XG4gICAgICAgICAgICAgICAgc3R5bGU9e3sgZGlzcGxheTogbmV3X2ltYWdlX2Rpc3BsYXkgfX1cbiAgICAgICAgICAgICAgICBvbkxvYWQ9eygpID0+IHtcbiAgICAgICAgICAgICAgICAgIHNldF9vbGRfaW1hZ2VfdXJsKG5ld19pbWFnZV91cmwpO1xuICAgICAgICAgICAgICAgICAgc2V0X3Nob3dfb2xkX2ltZyhmYWxzZSk7XG4gICAgICAgICAgICAgICAgfX1cbiAgICAgICAgICAgICAgICBhbHQ9e3Bsb3QubmFtZX1cbiAgICAgICAgICAgICAgICBzcmM9e25ld19pbWFnZV91cmx9XG4gICAgICAgICAgICAgICAgc2V0SW1hZ2VFcnJvcj17c2V0SW1hZ2VFcnJvcn1cbiAgICAgICAgICAgICAgICB3aWR0aD17cGFyYW1zX2Zvcl9hcGkud2lkdGh9XG4gICAgICAgICAgICAgICAgaGVpZ2h0PXtwYXJhbXNfZm9yX2FwaS5oZWlnaHR9XG4gICAgICAgICAgICAgIC8+XG4gICAgICAgICAgICAgIHsvKldoZW4gaW1hZ2VzIGlzIHVwZGF0aW5nLCB3ZSBnZXR0aW5nIGJsaW5raW5nIGVmZmVjdC4gXG4gICAgICAgICAgICAgICAgICAgIFdlIHRyeWluZyB0byBhdm9pZCBpdCB3aXRoIHNob3dpbmcgb2xkIGltYWdlIGluc3RlYWQgb2Ygbm90aGluZyAod2hlbiBhIG5ldyBpbWFnZSBpcyBqdXN0IHJlcXVlc3RpbmcgcHJvY2VzcylcbiAgICAgICAgICAgICAgICAgICAgT2xkIGltYWdlIGlzIGFuIGltYWdlIHdoaWNoIGlzIDIwIHNlYyBvbGRlciB0aGVuIHRoZSBuZXcgcmVxdWVzdGVkIG9uZS5cbiAgICAgICAgICAgICAgICAgICAgKi99XG4gICAgICAgICAgICAgIDxJbWFnZUZhbGxiYWNrXG4gICAgICAgICAgICAgICAgcmV0cnlUaW1lcz17M31cbiAgICAgICAgICAgICAgICBzdHlsZT17eyBkaXNwbGF5OiBvbGRfaW1hZ2VfZGlzcGxheSB9fVxuICAgICAgICAgICAgICAgIGFsdD17cGxvdC5uYW1lfVxuICAgICAgICAgICAgICAgIHNyYz17b2xkX2ltYWdlX3VybH1cbiAgICAgICAgICAgICAgICBzZXRJbWFnZUVycm9yPXtzZXRJbWFnZUVycm9yfVxuICAgICAgICAgICAgICAgIHdpZHRoPXsnYXV0byd9XG4gICAgICAgICAgICAgICAgaGVpZ2h0PXsnYXV0byd9XG4gICAgICAgICAgICAgIC8+XG4gICAgICAgICAgICA8Lz5cbiAgICAgICAgICApfVxuICAgICAgICA8L2Rpdj5cbiAgICAgICl9XG4gICAgPC8+XG4gICk7XG59O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==