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
      lineNumber: 69,
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
      lineNumber: 71,
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
      lineNumber: 83,
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
      lineNumber: 100,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy9wbG90L3Bsb3RJbWFnZS50c3giXSwibmFtZXMiOlsiUGxvdEltYWdlIiwiaW1hZ2VSZWYiLCJxdWVyeSIsImlzUGxvdFNlbGVjdGVkIiwicGFyYW1zX2Zvcl9hcGkiLCJwbG90VVJMIiwidXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiIsImJsaW5rIiwicGxvdCIsIlJlYWN0Iiwicm9vdF91cmwiLCJuZXdfaW1hZ2VfdXJsIiwic2V0X25ld19pbWFnZV91cmwiLCJvbGRfaW1hZ2VfdXJsIiwic2V0X29sZF9pbWFnZV91cmwiLCJzaG93X29sZF9pbWciLCJzZXRfc2hvd19vbGRfaW1nIiwiaW1hZ2VFcnJvciIsInNldEltYWdlRXJyb3IiLCJjdXN0b21pemVQcm9wcyIsImhlaWdodCIsIndpZHRoIiwicnVuX251bWJlciIsImRhdGFzZXRfbmFtZSIsImx1bWkiLCJub3JtYWxpemUiLCJvbGRfaW1hZ2VfZGlzcGxheSIsIm5ld19pbWFnZV9kaXNwbGF5IiwicmVtb3ZlUGxvdEZyb21SaWdodFNpZGUiLCJhZGRQbG90VG9SaWdodFNpZGUiLCJzY3JvbGwiLCJkaXNwbGF5IiwibmFtZSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBRUE7QUFNQTtBQUNBO0FBQ0E7QUFnQk8sSUFBTUEsU0FBUyxHQUFHLFNBQVpBLFNBQVksT0FTSDtBQUFBOztBQUFBLE1BUnBCQyxRQVFvQixRQVJwQkEsUUFRb0I7QUFBQSxNQVBwQkMsS0FPb0IsUUFQcEJBLEtBT29CO0FBQUEsTUFOcEJDLGNBTW9CLFFBTnBCQSxjQU1vQjtBQUFBLE1BTHBCQyxjQUtvQixRQUxwQkEsY0FLb0I7QUFBQSxNQUpwQkMsT0FJb0IsUUFKcEJBLE9BSW9CO0FBQUEsTUFIcEJDLHlCQUdvQixRQUhwQkEseUJBR29CO0FBQUEsTUFGcEJDLEtBRW9CLFFBRnBCQSxLQUVvQjtBQUFBLE1BRHBCQyxJ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QUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBbEJGLENBWEosQ0FKSixDQURGO0FBaURELENBeEZNOztHQUFNbEIsUzs7S0FBQUEsUyIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4xNmYxOWE0YTVkYWZiYjRmNzk3OC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xuXG5pbXBvcnQgeyByb290X3VybCB9IGZyb20gJy4uLy4uLy4uL2NvbmZpZy9jb25maWcnO1xuaW1wb3J0IHtcbiAgUGFyYW1zRm9yQXBpUHJvcHMsXG4gIFF1ZXJ5UHJvcHMsXG59IGZyb20gJy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcbmltcG9ydCB7IHVzZVVwZGF0ZUxpdmVNb2RlIH0gZnJvbSAnLi4vLi4vLi4vaG9va3MvdXNlVXBkYXRlSW5MaXZlTW9kZSc7XG5pbXBvcnQgeyBFcnJvck1lc3NhZ2UgfSBmcm9tICcuLi9lcnJvck1lc3NhZ2UnO1xuaW1wb3J0IHsgSW1hZ2VGYWxsYmFjayB9IGZyb20gJy4uL2ltYWdlRmFsbGJhY2snO1xuaW1wb3J0IHtcbiAgYWRkUGxvdFRvUmlnaHRTaWRlLFxuICByZW1vdmVQbG90RnJvbVJpZ2h0U2lkZSxcbn0gZnJvbSAnLi9zaW5nbGVQbG90L3V0aWxzJztcblxuaW50ZXJmYWNlIFBsb3RJbWFnZVByb3BzIHtcbiAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbjogbnVtYmVyO1xuICBwYXJhbXNfZm9yX2FwaTogUGFyYW1zRm9yQXBpUHJvcHM7XG4gIGJsaW5rOiBib29sZWFuO1xuICBwbG90OiBhbnk7XG4gIHBsb3RVUkw6IHN0cmluZztcbiAgaXNQbG90U2VsZWN0ZWQ/OiBib29sZWFuO1xuICBxdWVyeTogUXVlcnlQcm9wcztcbiAgaW1hZ2VSZWY/OiBhbnk7XG59XG5cbmV4cG9ydCBjb25zdCBQbG90SW1hZ2UgPSAoe1xuICBpbWFnZVJlZixcbiAgcXVlcnksXG4gIGlzUGxvdFNlbGVjdGVkLFxuICBwYXJhbXNfZm9yX2FwaSxcbiAgcGxvdFVSTCxcbiAgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbixcbiAgYmxpbmssXG4gIHBsb3QsXG59OiBQbG90SW1hZ2VQcm9wcykgPT4ge1xuICBjb25zdCBbbmV3X2ltYWdlX3VybCwgc2V0X25ld19pbWFnZV91cmxdID0gUmVhY3QudXNlU3RhdGUoXG4gICAgYCR7cm9vdF91cmx9JHtwbG90VVJMfTtub3RPbGRlclRoYW49JHt1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFufWBcbiAgKTtcbiAgY29uc3QgW29sZF9pbWFnZV91cmwsIHNldF9vbGRfaW1hZ2VfdXJsXSA9IFJlYWN0LnVzZVN0YXRlKFxuICAgIGAke3Jvb3RfdXJsfSR7cGxvdFVSTH07bm90T2xkZXJUaGFuPSR7dXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbn1gXG4gICk7XG5cbiAgY29uc3QgW3Nob3dfb2xkX2ltZywgc2V0X3Nob3dfb2xkX2ltZ10gPSBSZWFjdC51c2VTdGF0ZSh0cnVlKTtcbiAgY29uc3QgW2ltYWdlRXJyb3IsIHNldEltYWdlRXJyb3JdID0gUmVhY3QudXNlU3RhdGUoZmFsc2UpO1xuXG4gIFJlYWN0LnVzZUVmZmVjdCgoKSA9PiB7XG4gICAgc2V0X25ld19pbWFnZV91cmwoXG4gICAgICBgJHtyb290X3VybH0ke3Bsb3RVUkx9O25vdE9sZGVyVGhhbj0ke3VwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW59YFxuICAgICk7XG4gICAgc2V0X3Nob3dfb2xkX2ltZyhibGluayk7XG4gIH0sIFtcbiAgICB1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuLFxuICAgIHBhcmFtc19mb3JfYXBpLmN1c3RvbWl6ZVByb3BzLFxuICAgIHBhcmFtc19mb3JfYXBpLmhlaWdodCxcbiAgICBwYXJhbXNfZm9yX2FwaS53aWR0aCxcbiAgICBwYXJhbXNfZm9yX2FwaS5ydW5fbnVtYmVyLFxuICAgIHBhcmFtc19mb3JfYXBpLmRhdGFzZXRfbmFtZSxcbiAgICBwYXJhbXNfZm9yX2FwaS5sdW1pLFxuICAgIHBhcmFtc19mb3JfYXBpLm5vcm1hbGl6ZSxcbiAgXSk7XG5cbiAgY29uc3Qgb2xkX2ltYWdlX2Rpc3BsYXkgPSBzaG93X29sZF9pbWcgPyAnJyA6ICdub25lJztcbiAgY29uc3QgbmV3X2ltYWdlX2Rpc3BsYXkgPSBzaG93X29sZF9pbWcgPyAnbm9uZScgOiAnJztcblxuICByZXR1cm4gKFxuICAgIDw+XG4gICAgICB7aW1hZ2VFcnJvciA/IChcbiAgICAgICAgPEVycm9yTWVzc2FnZSAvPlxuICAgICAgKSA6IChcbiAgICAgICAgPGRpdlxuICAgICAgICAgIG9uQ2xpY2s9e2FzeW5jICgpID0+IHtcbiAgICAgICAgICAgIGlmIChpbWFnZVJlZikge1xuICAgICAgICAgICAgICBpc1Bsb3RTZWxlY3RlZFxuICAgICAgICAgICAgICAgID8gYXdhaXQgcmVtb3ZlUGxvdEZyb21SaWdodFNpZGUocXVlcnksIHBsb3QpXG4gICAgICAgICAgICAgICAgOiBhd2FpdCBhZGRQbG90VG9SaWdodFNpZGUocXVlcnksIHBsb3QpO1xuICAgICAgICAgICAgICBzY3JvbGwoaW1hZ2VSZWYpO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH19XG4gICAgICAgID5cbiAgICAgICAgICB7IWltYWdlRXJyb3IgJiYgKFxuICAgICAgICAgICAgPD5cbiAgICAgICAgICAgICAgPEltYWdlRmFsbGJhY2tcbiAgICAgICAgICAgICAgICByZXRyeVRpbWVzPXszfVxuICAgICAgICAgICAgICAgIHN0eWxlPXt7IGRpc3BsYXk6IG5ld19pbWFnZV9kaXNwbGF5IH19XG4gICAgICAgICAgICAgICAgb25Mb2FkPXsoKSA9PiB7XG4gICAgICAgICAgICAgICAgICBzZXRfb2xkX2ltYWdlX3VybChuZXdfaW1hZ2VfdXJsKTtcbiAgICAgICAgICAgICAgICAgIHNldF9zaG93X29sZF9pbWcoZmFsc2UpO1xuICAgICAgICAgICAgICAgIH19XG4gICAgICAgICAgICAgICAgYWx0PXtwbG90Lm5hbWV9XG4gICAgICAgICAgICAgICAgc3JjPXtuZXdfaW1hZ2VfdXJsfVxuICAgICAgICAgICAgICAgIHNldEltYWdlRXJyb3I9e3NldEltYWdlRXJyb3J9XG4gICAgICAgICAgICAgICAgd2lkdGg9e3BhcmFtc19mb3JfYXBpLndpZHRofVxuICAgICAgICAgICAgICAgIGhlaWdodD17cGFyYW1zX2Zvcl9hcGkuaGVpZ2h0fVxuICAgICAgICAgICAgICAvPlxuICAgICAgICAgICAgICB7LypXaGVuIGltYWdlcyBpcyB1cGRhdGluZywgd2UgZ2V0dGluZyBibGlua2luZyBlZmZlY3QuIFxuICAgICAgICAgICAgICAgICAgICBXZSB0cnlpbmcgdG8gYXZvaWQgaXQgd2l0aCBzaG93aW5nIG9sZCBpbWFnZSBpbnN0ZWFkIG9mIG5vdGhpbmcgKHdoZW4gYSBuZXcgaW1hZ2UgaXMganVzdCByZXF1ZXN0aW5nIHByb2Nlc3MpXG4gICAgICAgICAgICAgICAgICAgIE9sZCBpbWFnZSBpcyBhbiBpbWFnZSB3aGljaCBpcyAyMCBzZWMgb2xkZXIgdGhlbiB0aGUgbmV3IHJlcXVlc3RlZCBvbmUuXG4gICAgICAgICAgICAgICAgICAgICovfVxuICAgICAgICAgICAgICA8SW1hZ2VGYWxsYmFja1xuICAgICAgICAgICAgICAgIHJldHJ5VGltZXM9ezN9XG4gICAgICAgICAgICAgICAgc3R5bGU9e3sgZGlzcGxheTogb2xkX2ltYWdlX2Rpc3BsYXkgfX1cbiAgICAgICAgICAgICAgICBhbHQ9e3Bsb3QubmFtZX1cbiAgICAgICAgICAgICAgICBzcmM9e29sZF9pbWFnZV91cmx9XG4gICAgICAgICAgICAgICAgc2V0SW1hZ2VFcnJvcj17c2V0SW1hZ2VFcnJvcn1cbiAgICAgICAgICAgICAgICB3aWR0aD17J2F1dG8nfVxuICAgICAgICAgICAgICAgIGhlaWdodD17J2F1dG8nfVxuICAgICAgICAgICAgICAvPlxuICAgICAgICAgICAgPC8+XG4gICAgICAgICAgKX1cbiAgICAgICAgPC9kaXY+XG4gICAgICApfVxuICAgIDwvPlxuICApO1xufTtcbiJdLCJzb3VyY2VSb290IjoiIn0=