webpackHotUpdate_N_E("pages/index",{

/***/ "./components/initialPage/latestRuns.tsx":
/*!***********************************************!*\
  !*** ./components/initialPage/latestRuns.tsx ***!
  \***********************************************/
/*! exports provided: LatestRuns */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "LatestRuns", function() { return LatestRuns; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _hooks_useRequest__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../../hooks/useRequest */ "./hooks/useRequest.tsx");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../../config/config */ "./config/config.ts");
/* harmony import */ var _containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../containers/search/styledComponents */ "./containers/search/styledComponents.tsx");
/* harmony import */ var _containers_search_noResultsFound__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../containers/search/noResultsFound */ "./containers/search/noResultsFound.tsx");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _hooks_useNewer__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../hooks/useNewer */ "./hooks/useNewer.tsx");
/* harmony import */ var _liveModeButton__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../liveModeButton */ "./components/liveModeButton.tsx");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _latestRunsList__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ./latestRunsList */ "./components/initialPage/latestRunsList.tsx");
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/initialPage/latestRuns.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];











var LatestRuns = function LatestRuns() {
  _s();

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_0__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_5__["store"]),
      updated_by_not_older_than = _React$useContext.updated_by_not_older_than;

  var data_get_by_mount = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_1__["useRequest"])(Object(_config_config__WEBPACK_IMPORTED_MODULE_2__["get_the_latest_runs"])(updated_by_not_older_than), {}, []);
  var data_get_by_not_older_than_update = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_1__["useRequest"])(Object(_config_config__WEBPACK_IMPORTED_MODULE_2__["get_the_latest_runs"])(updated_by_not_older_than), {}, [updated_by_not_older_than]);
  var data = Object(_hooks_useNewer__WEBPACK_IMPORTED_MODULE_6__["useNewer"])(data_get_by_mount.data, data_get_by_not_older_than_update.data);
  var errors = Object(_hooks_useNewer__WEBPACK_IMPORTED_MODULE_6__["useNewer"])(data_get_by_mount.errors, data_get_by_not_older_than_update.errors);
  var isLoading = data_get_by_mount.isLoading;
  var latest_runs = data && data.runs.sort(function (a, b) {
    return a - b;
  });
  return __jsx(react__WEBPACK_IMPORTED_MODULE_0__["Fragment"], null, !isLoading && errors.length > 0 ? errors.map(function (error) {
    return __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_3__["StyledAlert"], {
      key: error,
      message: error,
      type: "error",
      showIcon: true,
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 50,
        columnNumber: 11
      }
    });
  }) : isLoading ? __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_3__["SpinnerWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 53,
      columnNumber: 9
    }
  }, __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_3__["Spinner"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 54,
      columnNumber: 11
    }
  })) : __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_3__["LatestRunsSection"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 57,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_8__["CustomDiv"], {
    display: "flex",
    justifycontent: "flex-end",
    width: "auto",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 58,
      columnNumber: 11
    }
  }, __jsx(_liveModeButton__WEBPACK_IMPORTED_MODULE_7__["LiveModeButton"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 59,
      columnNumber: 13
    }
  })), __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_3__["LatestRunsTtitle"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 61,
      columnNumber: 11
    }
  }, "The latest runs"), isLoading ? __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_3__["SpinnerWrapper"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 63,
      columnNumber: 13
    }
  }, __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_3__["Spinner"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 64,
      columnNumber: 15
    }
  })) : latest_runs && latest_runs.length === 0 && !isLoading && errors.length === 0 ? __jsx(_containers_search_noResultsFound__WEBPACK_IMPORTED_MODULE_4__["NoResultsFound"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 70,
      columnNumber: 13
    }
  }) : latest_runs && __jsx(_latestRunsList__WEBPACK_IMPORTED_MODULE_9__["LatestRunsList"], {
    latest_runs: latest_runs,
    mode: _config_config__WEBPACK_IMPORTED_MODULE_2__["functions_config"].mode,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 73,
      columnNumber: 15
    }
  })));
};

_s(LatestRuns, "+5BwgV2qipd2Cg3GsyOMKDpjz4w=", false, function () {
  return [_hooks_useRequest__WEBPACK_IMPORTED_MODULE_1__["useRequest"], _hooks_useRequest__WEBPACK_IMPORTED_MODULE_1__["useRequest"], _hooks_useNewer__WEBPACK_IMPORTED_MODULE_6__["useNewer"], _hooks_useNewer__WEBPACK_IMPORTED_MODULE_6__["useNewer"]];
});

_c = LatestRuns;

var _c;

$RefreshReg$(_c, "LatestRuns");

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

/***/ }),

/***/ "./components/liveModeButton.tsx":
/*!***************************************!*\
  !*** ./components/liveModeButton.tsx ***!
  \***************************************/
/*! exports provided: LiveModeButton */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "LiveModeButton", function() { return LiveModeButton; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ./styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/router */ "./node_modules/next/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../hooks/useUpdateInLiveMode */ "./hooks/useUpdateInLiveMode.tsx");
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/liveModeButton.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];





var liveModeHandler = function liveModeHandler(liveModeRun, liveModeDataset) {
  next_router__WEBPACK_IMPORTED_MODULE_2___default.a.push({
    pathname: '/',
    query: {
      run_number: liveModeRun,
      dataset_name: liveModeDataset,
      folder_path: 'Summary'
    }
  });
};

var LiveModeButton = function LiveModeButton() {
  _s();

  var liveModeDataset = '/Global/Online/ALL';
  var liveModeRun = '0';

  var _useUpdateLiveMode = Object(_hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_3__["useUpdateLiveMode"])(),
      set_update = _useUpdateLiveMode.set_update,
      update = _useUpdateLiveMode.update;

  return __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_1__["LiveButton"], {
    onClick: function onClick() {
      liveModeHandler(liveModeRun, liveModeDataset);
      set_update(true);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 24,
      columnNumber: 5
    }
  }, "Live Mode");
};

_s(LiveModeButton, "189QJ/1iUzf3mOSRqkfQHd5oY0g=", false, function () {
  return [_hooks_useUpdateInLiveMode__WEBPACK_IMPORTED_MODULE_3__["useUpdateLiveMode"]];
});

_c = LiveModeButton;

var _c;

$RefreshReg$(_c, "LiveModeButton");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9pbml0aWFsUGFnZS9sYXRlc3RSdW5zLnRzeCIsIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9saXZlTW9kZUJ1dHRvbi50c3giXSwibmFtZXMiOlsiTGF0ZXN0UnVucyIsIlJlYWN0Iiwic3RvcmUiLCJ1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuIiwiZGF0YV9nZXRfYnlfbW91bnQiLCJ1c2VSZXF1ZXN0IiwiZ2V0X3RoZV9sYXRlc3RfcnVucyIsImRhdGFfZ2V0X2J5X25vdF9vbGRlcl90aGFuX3VwZGF0ZSIsImRhdGEiLCJ1c2VOZXdlciIsImVycm9ycyIsImlzTG9hZGluZyIsImxhdGVzdF9ydW5zIiwicnVucyIsInNvcnQiLCJhIiwiYiIsImxlbmd0aCIsIm1hcCIsImVycm9yIiwiZnVuY3Rpb25zX2NvbmZpZyIsIm1vZGUiLCJsaXZlTW9kZUhhbmRsZXIiLCJsaXZlTW9kZVJ1biIsImxpdmVNb2RlRGF0YXNldCIsIlJvdXRlciIsInB1c2giLCJwYXRobmFtZSIsInF1ZXJ5IiwicnVuX251bWJlciIsImRhdGFzZXRfbmFtZSIsImZvbGRlcl9wYXRoIiwiTGl2ZU1vZGVCdXR0b24iLCJ1c2VVcGRhdGVMaXZlTW9kZSIsInNldF91cGRhdGUiLCJ1cGRhdGUiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUVBO0FBQ0E7QUFDQTtBQU9BO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBRU8sSUFBTUEsVUFBVSxHQUFHLFNBQWJBLFVBQWEsR0FBTTtBQUFBOztBQUFBLDBCQUNRQyxnREFBQSxDQUFpQkMsK0RBQWpCLENBRFI7QUFBQSxNQUN0QkMseUJBRHNCLHFCQUN0QkEseUJBRHNCOztBQUc5QixNQUFNQyxpQkFBaUIsR0FBR0Msb0VBQVUsQ0FDbENDLDBFQUFtQixDQUFDSCx5QkFBRCxDQURlLEVBRWxDLEVBRmtDLEVBR2xDLEVBSGtDLENBQXBD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WUFBUSxFQUFFLEdBREE7QUFFVkMsU0FBSyxFQUFFO0FBQ0xDLGdCQUFVLEVBQUVOLFdBRFA7QUFFTE8sa0JBQVksRUFBRU4sZUFGVDtBQUdMTyxpQkFBVyxFQUFFO0FBSFI7QUFGRyxHQUFaO0FBUUQsQ0FURDs7QUFXTyxJQUFNQyxjQUFjLEdBQUcsU0FBakJBLGNBQWlCLEdBQU07QUFBQTs7QUFDbEMsTUFBTVIsZUFBZSxHQUFHLG9CQUF4QjtBQUNBLE1BQU1ELFdBQVcsR0FBRyxHQUFwQjs7QUFGa0MsMkJBR0hVLG9GQUFpQixFQUhkO0FBQUEsTUFHMUJDLFVBSDBCLHNCQUcxQkEsVUFIMEI7QUFBQSxNQUdkQyxNQUhjLHNCQUdkQSxNQUhjOztBQUtsQyxTQUNFLE1BQUMsNERBQUQ7QUFDRSxXQUFPLEVBQUUsbUJBQU07QUFDYmIscUJBQWUsQ0FBQ0MsV0FBRCxFQUFjQyxlQUFkLENBQWY7QUFDQVUsZ0JBQVUsQ0FBQyxJQUFELENBQVY7QUFDRCxLQUpIO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsaUJBREY7QUFVRCxDQWZNOztHQUFNRixjO1VBR29CQyw0RTs7O0tBSHBCRCxjIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmMzOTdlMThmMTkwOTBkOWVmZmQwLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XHJcblxyXG5pbXBvcnQgeyB1c2VSZXF1ZXN0IH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlUmVxdWVzdCc7XHJcbmltcG9ydCB7IGdldF90aGVfbGF0ZXN0X3J1bnMgfSBmcm9tICcuLi8uLi9jb25maWcvY29uZmlnJztcclxuaW1wb3J0IHtcclxuICBTcGlubmVyV3JhcHBlcixcclxuICBTcGlubmVyLFxyXG4gIExhdGVzdFJ1bnNUdGl0bGUsXHJcbiAgTGF0ZXN0UnVuc1NlY3Rpb24sXHJcbiAgU3R5bGVkQWxlcnQsXHJcbn0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9zZWFyY2gvc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7IE5vUmVzdWx0c0ZvdW5kIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9zZWFyY2gvbm9SZXN1bHRzRm91bmQnO1xyXG5pbXBvcnQgeyBzdG9yZSB9IGZyb20gJy4uLy4uL2NvbnRleHRzL2xlZnRTaWRlQ29udGV4dCc7XHJcbmltcG9ydCB7IHVzZU5ld2VyIH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlTmV3ZXInO1xyXG5pbXBvcnQgeyBmdW5jdGlvbnNfY29uZmlnIH0gZnJvbSAnLi4vLi4vY29uZmlnL2NvbmZpZyc7XHJcbmltcG9ydCB7IExpdmVNb2RlQnV0dG9uIH0gZnJvbSAnLi4vbGl2ZU1vZGVCdXR0b24nO1xyXG5pbXBvcnQgeyBDdXN0b21EaXYgfSBmcm9tICcuLi9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHsgTGF0ZXN0UnVuc0xpc3QgfSBmcm9tICcuL2xhdGVzdFJ1bnNMaXN0JztcclxuXHJcbmV4cG9ydCBjb25zdCBMYXRlc3RSdW5zID0gKCkgPT4ge1xyXG4gIGNvbnN0IHsgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiB9ID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSk7XHJcblxyXG4gIGNvbnN0IGRhdGFfZ2V0X2J5X21vdW50ID0gdXNlUmVxdWVzdChcclxuICAgIGdldF90aGVfbGF0ZXN0X3J1bnModXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiksXHJcbiAgICB7fSxcclxuICAgIFtdXHJcbiAgKTtcclxuXHJcbiAgY29uc3QgZGF0YV9nZXRfYnlfbm90X29sZGVyX3RoYW5fdXBkYXRlID0gdXNlUmVxdWVzdChcclxuICAgIGdldF90aGVfbGF0ZXN0X3J1bnModXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiksXHJcbiAgICB7fSxcclxuICAgIFt1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuXVxyXG4gICk7XHJcblxyXG4gIGNvbnN0IGRhdGEgPSB1c2VOZXdlcihcclxuICAgIGRhdGFfZ2V0X2J5X21vdW50LmRhdGEsXHJcbiAgICBkYXRhX2dldF9ieV9ub3Rfb2xkZXJfdGhhbl91cGRhdGUuZGF0YVxyXG4gICk7XHJcbiAgY29uc3QgZXJyb3JzID0gdXNlTmV3ZXIoXHJcbiAgICBkYXRhX2dldF9ieV9tb3VudC5lcnJvcnMsXHJcbiAgICBkYXRhX2dldF9ieV9ub3Rfb2xkZXJfdGhhbl91cGRhdGUuZXJyb3JzXHJcbiAgKTtcclxuICBjb25zdCBpc0xvYWRpbmcgPSBkYXRhX2dldF9ieV9tb3VudC5pc0xvYWRpbmc7XHJcbiAgY29uc3QgbGF0ZXN0X3J1bnMgPSBkYXRhICYmIGRhdGEucnVucy5zb3J0KChhOiBudW1iZXIsIGI6IG51bWJlcikgPT4gYSAtIGIpO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPD5cclxuICAgICAgeyFpc0xvYWRpbmcgJiYgZXJyb3JzLmxlbmd0aCA+IDAgPyAoXHJcbiAgICAgICAgZXJyb3JzLm1hcCgoZXJyb3I6IHN0cmluZykgPT4gKFxyXG4gICAgICAgICAgPFN0eWxlZEFsZXJ0IGtleT17ZXJyb3J9IG1lc3NhZ2U9e2Vycm9yfSB0eXBlPVwiZXJyb3JcIiBzaG93SWNvbiAvPlxyXG4gICAgICAgICkpXHJcbiAgICAgICkgOiBpc0xvYWRpbmcgPyAoXHJcbiAgICAgICAgPFNwaW5uZXJXcmFwcGVyPlxyXG4gICAgICAgICAgPFNwaW5uZXIgLz5cclxuICAgICAgICA8L1NwaW5uZXJXcmFwcGVyPlxyXG4gICAgICApIDogKFxyXG4gICAgICAgIDxMYXRlc3RSdW5zU2VjdGlvbj5cclxuICAgICAgICAgIDxDdXN0b21EaXYgZGlzcGxheT1cImZsZXhcIiBqdXN0aWZ5Y29udGVudD1cImZsZXgtZW5kXCIgd2lkdGg9XCJhdXRvXCI+XHJcbiAgICAgICAgICAgIDxMaXZlTW9kZUJ1dHRvbiAvPlxyXG4gICAgICAgICAgPC9DdXN0b21EaXY+XHJcbiAgICAgICAgICA8TGF0ZXN0UnVuc1R0aXRsZT5UaGUgbGF0ZXN0IHJ1bnM8L0xhdGVzdFJ1bnNUdGl0bGU+XHJcbiAgICAgICAgICB7aXNMb2FkaW5nID8gKFxyXG4gICAgICAgICAgICA8U3Bpbm5lcldyYXBwZXI+XHJcbiAgICAgICAgICAgICAgPFNwaW5uZXIgLz5cclxuICAgICAgICAgICAgPC9TcGlubmVyV3JhcHBlcj5cclxuICAgICAgICAgICkgOiBsYXRlc3RfcnVucyAmJlxyXG4gICAgICAgICAgICBsYXRlc3RfcnVucy5sZW5ndGggPT09IDAgJiZcclxuICAgICAgICAgICAgIWlzTG9hZGluZyAmJlxyXG4gICAgICAgICAgICBlcnJvcnMubGVuZ3RoID09PSAwID8gKFxyXG4gICAgICAgICAgICA8Tm9SZXN1bHRzRm91bmQgLz5cclxuICAgICAgICAgICkgOiAoXHJcbiAgICAgICAgICAgIGxhdGVzdF9ydW5zICYmIChcclxuICAgICAgICAgICAgICA8TGF0ZXN0UnVuc0xpc3RcclxuICAgICAgICAgICAgICAgIGxhdGVzdF9ydW5zPXtsYXRlc3RfcnVuc31cclxuICAgICAgICAgICAgICAgIG1vZGU9e2Z1bmN0aW9uc19jb25maWcubW9kZX1cclxuICAgICAgICAgICAgICAvPlxyXG4gICAgICAgICAgICApXHJcbiAgICAgICAgICApfVxyXG4gICAgICAgIDwvTGF0ZXN0UnVuc1NlY3Rpb24+XHJcbiAgICAgICl9XHJcbiAgICA8Lz5cclxuICApO1xyXG59O1xyXG4iLCJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XHJcblxyXG5pbXBvcnQgeyBMaXZlQnV0dG9uIH0gZnJvbSAnLi9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IFJvdXRlciBmcm9tICduZXh0L3JvdXRlcic7XHJcbmltcG9ydCB7IHVzZVVwZGF0ZUxpdmVNb2RlIH0gZnJvbSAnLi4vaG9va3MvdXNlVXBkYXRlSW5MaXZlTW9kZSc7XHJcblxyXG5jb25zdCBsaXZlTW9kZUhhbmRsZXIgPSAobGl2ZU1vZGVSdW46IHN0cmluZywgbGl2ZU1vZGVEYXRhc2V0OiBzdHJpbmcpID0+IHtcclxuICBSb3V0ZXIucHVzaCh7XHJcbiAgICBwYXRobmFtZTogJy8nLFxyXG4gICAgcXVlcnk6IHtcclxuICAgICAgcnVuX251bWJlcjogbGl2ZU1vZGVSdW4sXHJcbiAgICAgIGRhdGFzZXRfbmFtZTogbGl2ZU1vZGVEYXRhc2V0LFxyXG4gICAgICBmb2xkZXJfcGF0aDogJ1N1bW1hcnknLFxyXG4gICAgfSxcclxuICB9KTtcclxufTtcclxuXHJcbmV4cG9ydCBjb25zdCBMaXZlTW9kZUJ1dHRvbiA9ICgpID0+IHtcclxuICBjb25zdCBsaXZlTW9kZURhdGFzZXQgPSAnL0dsb2JhbC9PbmxpbmUvQUxMJztcclxuICBjb25zdCBsaXZlTW9kZVJ1biA9ICcwJztcclxuICBjb25zdCB7IHNldF91cGRhdGUsIHVwZGF0ZSB9ID0gdXNlVXBkYXRlTGl2ZU1vZGUoKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxMaXZlQnV0dG9uXHJcbiAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICBsaXZlTW9kZUhhbmRsZXIobGl2ZU1vZGVSdW4sIGxpdmVNb2RlRGF0YXNldCk7XHJcbiAgICAgICAgc2V0X3VwZGF0ZSh0cnVlKTtcclxuICAgICAgfX1cclxuICAgID5cclxuICAgICAgTGl2ZSBNb2RlXHJcbiAgICA8L0xpdmVCdXR0b24+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==