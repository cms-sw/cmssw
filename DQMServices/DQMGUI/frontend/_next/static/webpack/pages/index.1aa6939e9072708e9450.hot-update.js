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

  react__WEBPACK_IMPORTED_MODULE_0__["useEffect"](function () {
    // set_update(true);
    return function () {
      return console.log('rertun');
    };
  }, []);
  return __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_2__["LatestRunsWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 27,
      columnNumber: 5
    }
  }, latest_runs.map(function (run) {
    return __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledCol"], {
      key: run.toString(),
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 29,
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
        lineNumber: 30,
        columnNumber: 11
      }
    }, __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_2__["StyledA"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 39,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9pbml0aWFsUGFnZS9sYXRlc3RSdW5zTGlzdC50c3giXSwibmFtZXMiOlsiTGF0ZXN0UnVuc0xpc3QiLCJsYXRlc3RfcnVucyIsIm1vZGUiLCJ1c2VCbGlua09uVXBkYXRlIiwiYmxpbmsiLCJ1c2VVcGRhdGVMaXZlTW9kZSIsInNldF91cGRhdGUiLCJSZWFjdCIsImNvbnNvbGUiLCJsb2ciLCJtYXAiLCJydW4iLCJ0b1N0cmluZyIsImNoYW5nZVJvdXRlciIsInNlYXJjaF9ydW5fbnVtYmVyIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQU1BO0FBQ0E7QUFPTyxJQUFNQSxjQUFjLEdBQUcsU0FBakJBLGNBQWlCLE9BQWdEO0FBQUE7O0FBQUEsTUFBN0NDLFdBQTZDLFFBQTdDQSxXQUE2QztBQUFBLE1BQWhDQyxJQUFnQyxRQUFoQ0EsSUFBZ0M7O0FBQUEsMEJBQzFEQyxnRkFBZ0IsRUFEMEM7QUFBQSxNQUNwRUMsS0FEb0UscUJBQ3BFQSxLQURvRTs7QUFBQSwyQkFHckRDLG9GQUFpQixFQUhvQztBQUFBLE1BR3BFQyxVQUhvRSxzQkFHcEVBLFVBSG9FOztBQUk1RUMsaURBQUEsQ0FBZ0IsWUFBTTtBQUNwQjtBQUNBLFdBQU87QUFBQSxhQUFLQyxPQUFPLENBQUNDLEdBQVIsQ0FBWSxRQUFaLENBQUw7QUFBQSxLQUFQO0FBQ0QsR0FIRCxFQUdHLEVBSEg7QUFLQSxTQUNFLE1BQUMscUZBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNHUixXQUFXLENBQUNTLEdBQVosQ0FBZ0IsVUFBQ0MsR0FBRDtBQUFBLFdBQ2YsTUFBQyw2RUFBRDtBQUFXLFNBQUcsRUFBRUEsR0FBRyxDQUFDQyxRQUFKLEVBQWhCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FDRSxNQUFDLDhFQUFEO0FBQ0UsZUFBUyxFQUFFUixLQUFLLENBQUNRLFFBQU4sRUFEYjtBQUVFLGVBQVMsRUFBRSxDQUFDVixJQUFJLEtBQUssUUFBVixFQUFvQlUsUUFBcEIsRUFGYjtBQUdFLFdBQUssRUFBQyxNQUhSO0FBSUUsYUFBTyxFQUFFLG1CQUFNO0FBQ2JOLGtCQUFVLENBQUMsS0FBRCxDQUFWO0FBQ0FPLHNGQUFZLENBQUM7QUFBRUMsMkJBQWlCLEVBQUVIO0FBQXJCLFNBQUQsQ0FBWjtBQUNELE9BUEg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQVNFLE1BQUMsMkVBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUFVQSxHQUFWLENBVEYsQ0FERixDQURlO0FBQUEsR0FBaEIsQ0FESCxDQURGO0FBbUJELENBNUJNOztHQUFNWCxjO1VBQ09HLHdFLEVBRUtFLDRFOzs7S0FIWkwsYyIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC4xYWE2OTM5ZTkwNzI3MDhlOTQ1MC5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xyXG5pbXBvcnQgeyBjaGFuZ2VSb3V0ZXIgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvdXRpbHMnO1xyXG5pbXBvcnQge1xyXG4gIExhdGVzdFJ1bnNXcmFwcGVyLFxyXG4gIFJ1bldyYXBwZXIsXHJcbiAgU3R5bGVkQSxcclxuICBTdHlsZWRDb2wsXHJcbn0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9zZWFyY2gvc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7IHVzZUJsaW5rT25VcGRhdGUgfSBmcm9tICcuLi8uLi9ob29rcy91c2VCbGlua09uVXBkYXRlJztcclxuaW1wb3J0IHsgdXNlVXBkYXRlTGl2ZU1vZGUgfSBmcm9tICcuLi8uLi9ob29rcy91c2VVcGRhdGVJbkxpdmVNb2RlJztcclxuXHJcbmludGVyZmFjZSBMYXRlc3RSdW5zTGlzdFByb3BzIHtcclxuICBsYXRlc3RfcnVuczogbnVtYmVyW107XHJcbiAgbW9kZTogc3RyaW5nO1xyXG59XHJcblxyXG5leHBvcnQgY29uc3QgTGF0ZXN0UnVuc0xpc3QgPSAoeyBsYXRlc3RfcnVucywgbW9kZSB9OiBMYXRlc3RSdW5zTGlzdFByb3BzKSA9PiB7XHJcbiAgY29uc3QgeyBibGluayB9ID0gdXNlQmxpbmtPblVwZGF0ZSgpO1xyXG5cclxuICBjb25zdCB7IHNldF91cGRhdGUgfSA9IHVzZVVwZGF0ZUxpdmVNb2RlKCk7XHJcbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcclxuICAgIC8vIHNldF91cGRhdGUodHJ1ZSk7XHJcbiAgICByZXR1cm4gKCk9PiBjb25zb2xlLmxvZygncmVydHVuJylcclxuICB9LCBbXSk7XHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8TGF0ZXN0UnVuc1dyYXBwZXI+XHJcbiAgICAgIHtsYXRlc3RfcnVucy5tYXAoKHJ1bjogbnVtYmVyKSA9PiAoXHJcbiAgICAgICAgPFN0eWxlZENvbCBrZXk9e3J1bi50b1N0cmluZygpfT5cclxuICAgICAgICAgIDxSdW5XcmFwcGVyXHJcbiAgICAgICAgICAgIGlzTG9hZGluZz17YmxpbmsudG9TdHJpbmcoKX1cclxuICAgICAgICAgICAgYW5pbWF0aW9uPXsobW9kZSA9PT0gJ09OTElORScpLnRvU3RyaW5nKCl9XHJcbiAgICAgICAgICAgIGhvdmVyPVwidHJ1ZVwiXHJcbiAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICBzZXRfdXBkYXRlKGZhbHNlKTtcclxuICAgICAgICAgICAgICBjaGFuZ2VSb3V0ZXIoeyBzZWFyY2hfcnVuX251bWJlcjogcnVuIH0pO1xyXG4gICAgICAgICAgICB9fVxyXG4gICAgICAgICAgPlxyXG4gICAgICAgICAgICA8U3R5bGVkQT57cnVufTwvU3R5bGVkQT5cclxuICAgICAgICAgIDwvUnVuV3JhcHBlcj5cclxuICAgICAgICA8L1N0eWxlZENvbD5cclxuICAgICAgKSl9XHJcbiAgICA8L0xhdGVzdFJ1bnNXcmFwcGVyPlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=