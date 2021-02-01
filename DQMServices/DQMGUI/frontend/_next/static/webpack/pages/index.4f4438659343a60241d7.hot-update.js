webpackHotUpdate_N_E("pages/index",{

/***/ "./components/initialPage/latestRunsList.tsx":
/*!***************************************************!*\
  !*** ./components/initialPage/latestRunsList.tsx ***!
  \***************************************************/
/*! exports provided: LatestRunsList */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "LatestRunsList", function() { return LatestRunsList; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../../containers/display/utils */ "./containers/display/utils.ts");
/* harmony import */ var _containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../../containers/search/styledComponents */ "./containers/search/styledComponents.tsx");
/* harmony import */ var _hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../hooks/useBlinkOnUpdate */ "./hooks/useBlinkOnUpdate.tsx");
/* harmony import */ var _hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../hooks/useUpdateInLiveMode */ "./hooks/useUpdateInLiveMode.tsx");
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/initialPage/latestRunsList.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];





var LatestRunsList = function LatestRunsList(_ref) {
  _s();

  var latest_runs = _ref.latest_runs,
      mode = _ref.mode;

  var _useBlinkOnUpdate = Object(_hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_3__["useBlinkOnUpdate"])(),
      blink = _useBlinkOnUpdate.blink;

  var _useUpdateLiveMode = Object(_hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_4__["useUpdateLiveMode"])(),
      set_update = _useUpdateLiveMode.set_update;

  react__WEBPACK_IMPORTED_MODULE_0__["useEffect"](function () {// set_update(true);
  }, []);
  return __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_2__["LatestRunsWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 26,
      columnNumber: 5
    }
  }, latest_runs.map(function (run) {
    return __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledCol"], {
      key: run.toString(),
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 28,
        columnNumber: 9
      }
    }, __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_2__["RunWrapper"], {
      isLoading: blink.toString(),
      animation: (mode === 'ONLINE').toString(),
      hover: "true",
      onClick: function onClick() {
        set_update(false);
        Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_1__["changeRouter"])({
          search_run_number: run
        });
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 29,
        columnNumber: 11
      }
    }, __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledA"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 38,
        columnNumber: 13
      }
    }, run)));
  }));
};

_s(LatestRunsList, "n3sKbc8fd6YBpqefg9Yy/V8VSvo=", false, function () {
  return [_hooks_useBlinkOnUpdate__WEBPACK_IMPORTED_MODULE_3__["useBlinkOnUpdate"], _hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_4__["useUpdateLiveMode"]];
});

_c = LatestRunsList;

var _c;

$RefreshReg$(_c, "LatestRunsList");

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

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9pbml0aWFsUGFnZS9sYXRlc3RSdW5zTGlzdC50c3giXSwibmFtZXMiOlsiTGF0ZXN0UnVuc0xpc3QiLCJsYXRlc3RfcnVucyIsIm1vZGUiLCJ1c2VCbGlua09uVXBkYXRlIiwiYmxpbmsiLCJ1c2VVcGRhdGVMaXZlTW9kZSIsInNldF91cGRhdGUiLCJSZWFjdCIsIm1hcCIsInJ1biIsInRvU3RyaW5nIiwiY2hhbmdlUm91dGVyIiwic2VhcmNoX3J1bl9udW1iZXIiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBTUE7QUFDQTtBQU9PLElBQU1BLGNBQWMsR0FBRyxTQUFqQkEsY0FBaUIsT0FBZ0Q7QUFBQTs7QUFBQSxNQUE3Q0MsV0FBNkMsUUFBN0NBLFdBQTZDO0FBQUEsTUFBaENDLElBQWdDLFFBQWhDQSxJQUFnQzs7QUFBQSwwQkFDMURDLGdGQUFnQixFQUQwQztBQUFBLE1BQ3BFQyxLQURvRSxxQkFDcEVBLEtBRG9FOztBQUFBLDJCQUdyREMsb0ZBQWlCLEVBSG9DO0FBQUEsTUFHcEVDLFVBSG9FLHNCQUdwRUEsVUFIb0U7O0FBSTVFQyxpREFBQSxDQUFnQixZQUFNLENBQ3BCO0FBQ0QsR0FGRCxFQUVHLEVBRkg7QUFJQSxTQUNFLE1BQUMscUZBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNHTixXQUFXLENBQUNPLEdBQVosQ0FBZ0IsVUFBQ0MsR0FBRDtBQUFBLFdBQ2YsTUFBQyw2RUFBRDtBQUFXLFNBQUcsRUFBRUEsR0FBRyxDQUFDQyxRQUFKLEVBQWhCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FDRSxNQUFDLDhFQUFEO0FBQ0UsZUFBUyxFQUFFTixLQUFLLENBQUNNLFFBQU4sRUFEYjtBQUVFLGVBQVMsRUFBRSxDQUFDUixJQUFJLEtBQUssUUFBVixFQUFvQlEsUUFBcEIsRUFGYjtBQUdFLFdBQUssRUFBQyxNQUhSO0FBSUUsYUFBTyxFQUFFLG1CQUFNO0FBQ2JKLGtCQUFVLENBQUMsS0FBRCxDQUFWO0FBQ0FLLHNGQUFZLENBQUM7QUFBRUMsMkJBQWlCLEVBQUVIO0FBQXJCLFNBQUQsQ0FBWjtBQUNELE9BUEg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQVNFLE1BQUMsMkVBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUFVQSxHQUFWLENBVEYsQ0FERixDQURlO0FBQUEsR0FBaEIsQ0FESCxDQURGO0FBbUJELENBM0JNOztHQUFNVCxjO1VBQ09HLHdFLEVBRUtFLDRFOzs7S0FIWkwsYyIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC40ZjQ0Mzg2NTkzNDNhNjAyNDFkNy5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xyXG5pbXBvcnQgeyBjaGFuZ2VSb3V0ZXIgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvdXRpbHMnO1xyXG5pbXBvcnQge1xyXG4gIExhdGVzdFJ1bnNXcmFwcGVyLFxyXG4gIFJ1bldyYXBwZXIsXHJcbiAgU3R5bGVkQSxcclxuICBTdHlsZWRDb2wsXHJcbn0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9zZWFyY2gvc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7IHVzZUJsaW5rT25VcGRhdGUgfSBmcm9tICcuLi8uLi9ob29rcy91c2VCbGlua09uVXBkYXRlJztcclxuaW1wb3J0IHsgdXNlVXBkYXRlTGl2ZU1vZGUgfSBmcm9tICcuLi8uLi9ob29rcy91c2VVcGRhdGVJbkxpdmVNb2RlJztcclxuXHJcbmludGVyZmFjZSBMYXRlc3RSdW5zTGlzdFByb3BzIHtcclxuICBsYXRlc3RfcnVuczogbnVtYmVyW107XHJcbiAgbW9kZTogc3RyaW5nO1xyXG59XHJcblxyXG5leHBvcnQgY29uc3QgTGF0ZXN0UnVuc0xpc3QgPSAoeyBsYXRlc3RfcnVucywgbW9kZSB9OiBMYXRlc3RSdW5zTGlzdFByb3BzKSA9PiB7XHJcbiAgY29uc3QgeyBibGluayB9ID0gdXNlQmxpbmtPblVwZGF0ZSgpO1xyXG5cclxuICBjb25zdCB7IHNldF91cGRhdGUgfSA9IHVzZVVwZGF0ZUxpdmVNb2RlKCk7XHJcbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcclxuICAgIC8vIHNldF91cGRhdGUodHJ1ZSk7XHJcbiAgfSwgW10pO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPExhdGVzdFJ1bnNXcmFwcGVyPlxyXG4gICAgICB7bGF0ZXN0X3J1bnMubWFwKChydW46IG51bWJlcikgPT4gKFxyXG4gICAgICAgIDxTdHlsZWRDb2wga2V5PXtydW4udG9TdHJpbmcoKX0+XHJcbiAgICAgICAgICA8UnVuV3JhcHBlclxyXG4gICAgICAgICAgICBpc0xvYWRpbmc9e2JsaW5rLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgICAgIGFuaW1hdGlvbj17KG1vZGUgPT09ICdPTkxJTkUnKS50b1N0cmluZygpfVxyXG4gICAgICAgICAgICBob3Zlcj1cInRydWVcIlxyXG4gICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgc2V0X3VwZGF0ZShmYWxzZSk7XHJcbiAgICAgICAgICAgICAgY2hhbmdlUm91dGVyKHsgc2VhcmNoX3J1bl9udW1iZXI6IHJ1biB9KTtcclxuICAgICAgICAgICAgfX1cclxuICAgICAgICAgID5cclxuICAgICAgICAgICAgPFN0eWxlZEE+e3J1bn08L1N0eWxlZEE+XHJcbiAgICAgICAgICA8L1J1bldyYXBwZXI+XHJcbiAgICAgICAgPC9TdHlsZWRDb2w+XHJcbiAgICAgICkpfVxyXG4gICAgPC9MYXRlc3RSdW5zV3JhcHBlcj5cclxuICApO1xyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9