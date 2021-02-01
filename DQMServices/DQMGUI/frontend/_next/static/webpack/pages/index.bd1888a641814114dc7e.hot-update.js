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
    return console.log('rertun');
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9pbml0aWFsUGFnZS9sYXRlc3RSdW5zTGlzdC50c3giXSwibmFtZXMiOlsiTGF0ZXN0UnVuc0xpc3QiLCJsYXRlc3RfcnVucyIsIm1vZGUiLCJ1c2VCbGlua09uVXBkYXRlIiwiYmxpbmsiLCJ1c2VVcGRhdGVMaXZlTW9kZSIsInNldF91cGRhdGUiLCJSZWFjdCIsImNvbnNvbGUiLCJsb2ciLCJtYXAiLCJydW4iLCJ0b1N0cmluZyIsImNoYW5nZVJvdXRlciIsInNlYXJjaF9ydW5fbnVtYmVyIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQU1BO0FBQ0E7QUFPTyxJQUFNQSxjQUFjLEdBQUcsU0FBakJBLGNBQWlCLE9BQWdEO0FBQUE7O0FBQUEsTUFBN0NDLFdBQTZDLFFBQTdDQSxXQUE2QztBQUFBLE1BQWhDQyxJQUFnQyxRQUFoQ0EsSUFBZ0M7O0FBQUEsMEJBQzFEQyxnRkFBZ0IsRUFEMEM7QUFBQSxNQUNwRUMsS0FEb0UscUJBQ3BFQSxLQURvRTs7QUFBQSwyQkFHckRDLG9GQUFpQixFQUhvQztBQUFBLE1BR3BFQyxVQUhvRSxzQkFHcEVBLFVBSG9FOztBQUk1RUMsaURBQUEsQ0FBZ0IsWUFBTTtBQUNwQjtBQUNBLFdBQU9DLE9BQU8sQ0FBQ0MsR0FBUixDQUFZLFFBQVosQ0FBUDtBQUNELEdBSEQsRUFHRyxFQUhIO0FBS0EsU0FDRSxNQUFDLHFGQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDR1IsV0FBVyxDQUFDUyxHQUFaLENBQWdCLFVBQUNDLEdBQUQ7QUFBQSxXQUNmLE1BQUMsNkVBQUQ7QUFBVyxTQUFHLEVBQUVBLEdBQUcsQ0FBQ0MsUUFBSixFQUFoQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQ0UsTUFBQyw4RUFBRDtBQUNFLGVBQVMsRUFBRVIsS0FBSyxDQUFDUSxRQUFOLEVBRGI7QUFFRSxlQUFTLEVBQUUsQ0FBQ1YsSUFBSSxLQUFLLFFBQVYsRUFBb0JVLFFBQXBCLEVBRmI7QUFHRSxXQUFLLEVBQUMsTUFIUjtBQUlFLGFBQU8sRUFBRSxtQkFBTTtBQUNiTixrQkFBVSxDQUFDLEtBQUQsQ0FBVjtBQUNBTyxzRkFBWSxDQUFDO0FBQUVDLDJCQUFpQixFQUFFSDtBQUFyQixTQUFELENBQVo7QUFDRCxPQVBIO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FTRSxNQUFDLDJFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FBVUEsR0FBVixDQVRGLENBREYsQ0FEZTtBQUFBLEdBQWhCLENBREgsQ0FERjtBQW1CRCxDQTVCTTs7R0FBTVgsYztVQUNPRyx3RSxFQUVLRSw0RTs7O0tBSFpMLGMiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguYmQxODg4YTY0MTgxNDExNGRjN2UuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgY2hhbmdlUm91dGVyIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3V0aWxzJztcclxuaW1wb3J0IHtcclxuICBMYXRlc3RSdW5zV3JhcHBlcixcclxuICBSdW5XcmFwcGVyLFxyXG4gIFN0eWxlZEEsXHJcbiAgU3R5bGVkQ29sLFxyXG59IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvc2VhcmNoL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyB1c2VCbGlua09uVXBkYXRlIH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlQmxpbmtPblVwZGF0ZSc7XHJcbmltcG9ydCB7IHVzZVVwZGF0ZUxpdmVNb2RlIH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlVXBkYXRlSW5MaXZlTW9kZSc7XHJcblxyXG5pbnRlcmZhY2UgTGF0ZXN0UnVuc0xpc3RQcm9wcyB7XHJcbiAgbGF0ZXN0X3J1bnM6IG51bWJlcltdO1xyXG4gIG1vZGU6IHN0cmluZztcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IExhdGVzdFJ1bnNMaXN0ID0gKHsgbGF0ZXN0X3J1bnMsIG1vZGUgfTogTGF0ZXN0UnVuc0xpc3RQcm9wcykgPT4ge1xyXG4gIGNvbnN0IHsgYmxpbmsgfSA9IHVzZUJsaW5rT25VcGRhdGUoKTtcclxuXHJcbiAgY29uc3QgeyBzZXRfdXBkYXRlIH0gPSB1c2VVcGRhdGVMaXZlTW9kZSgpO1xyXG4gIFJlYWN0LnVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICAvLyBzZXRfdXBkYXRlKHRydWUpO1xyXG4gICAgcmV0dXJuIGNvbnNvbGUubG9nKCdyZXJ0dW4nKVxyXG4gIH0sIFtdKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxMYXRlc3RSdW5zV3JhcHBlcj5cclxuICAgICAge2xhdGVzdF9ydW5zLm1hcCgocnVuOiBudW1iZXIpID0+IChcclxuICAgICAgICA8U3R5bGVkQ29sIGtleT17cnVuLnRvU3RyaW5nKCl9PlxyXG4gICAgICAgICAgPFJ1bldyYXBwZXJcclxuICAgICAgICAgICAgaXNMb2FkaW5nPXtibGluay50b1N0cmluZygpfVxyXG4gICAgICAgICAgICBhbmltYXRpb249eyhtb2RlID09PSAnT05MSU5FJykudG9TdHJpbmcoKX1cclxuICAgICAgICAgICAgaG92ZXI9XCJ0cnVlXCJcclxuICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgIHNldF91cGRhdGUoZmFsc2UpO1xyXG4gICAgICAgICAgICAgIGNoYW5nZVJvdXRlcih7IHNlYXJjaF9ydW5fbnVtYmVyOiBydW4gfSk7XHJcbiAgICAgICAgICAgIH19XHJcbiAgICAgICAgICA+XHJcbiAgICAgICAgICAgIDxTdHlsZWRBPntydW59PC9TdHlsZWRBPlxyXG4gICAgICAgICAgPC9SdW5XcmFwcGVyPlxyXG4gICAgICAgIDwvU3R5bGVkQ29sPlxyXG4gICAgICApKX1cclxuICAgIDwvTGF0ZXN0UnVuc1dyYXBwZXI+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==