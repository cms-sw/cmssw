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
/* harmony import */ var _latestRunsList__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ./latestRunsList */ "./components/initialPage/latestRunsList.tsx");
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
  }, __jsx(_containers_search_styledComponents__WEBPACK_IMPORTED_MODULE_3__["LatestRunsTtitle"], {
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
  }) : latest_runs && __jsx(_latestRunsList__WEBPACK_IMPORTED_MODULE_7__["LatestRunsList"], {
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
false

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9pbml0aWFsUGFnZS9sYXRlc3RSdW5zLnRzeCJdLCJuYW1lcyI6WyJMYXRlc3RSdW5zIiwiUmVhY3QiLCJzdG9yZSIsInVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4iLCJkYXRhX2dldF9ieV9tb3VudCIsInVzZVJlcXVlc3QiLCJnZXRfdGhlX2xhdGVzdF9ydW5zIiwiZGF0YV9nZXRfYnlfbm90X29sZGVyX3RoYW5fdXBkYXRlIiwiZGF0YSIsInVzZU5ld2VyIiwiZXJyb3JzIiwiaXNMb2FkaW5nIiwibGF0ZXN0X3J1bnMiLCJydW5zIiwic29ydCIsImEiLCJiIiwibGVuZ3RoIiwibWFwIiwiZXJyb3IiLCJmdW5jdGlvbnNfY29uZmlnIiwibW9kZSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUVBO0FBQ0E7QUFDQTtBQU9BO0FBQ0E7QUFDQTtBQUNBO0FBR0E7QUFFTyxJQUFNQSxVQUFVLEdBQUcsU0FBYkEsVUFBYSxHQUFNO0FBQUE7O0FBQUEsMEJBQ1FDLGdEQUFBLENBQWlCQywrREFBakIsQ0FEUjtBQUFBLE1BQ3RCQyx5QkFEc0IscUJBQ3RCQSx5QkFEc0I7O0FBRzlCLE1BQU1DLGlCQUFpQixHQUFHQyxvRUFBVSxDQUNsQ0MsMEVBQW1CLENBQUNILHlCQUFELENBRGUsRUFFbEMsRUFGa0MsRUFHbEMsRUFIa0MsQ0FBcEM7QUFNQSxNQUFNSSxpQ0FBaUMsR0FBR0Ysb0VBQVUsQ0FDbERDLDBFQUFtQixDQUFDSCx5QkFBRCxDQUQrQixFQUVsRCxFQUZrRCxFQUdsRCxDQUFDQSx5QkFBRCxDQUhrRCxDQUFwRDtBQU1BLE1BQU1LLElBQUksR0FBR0MsZ0VBQVEsQ0FDbkJMLGlCQUFpQixDQUFDSSxJQURDLEVBRW5CRCxpQ0FBaUMsQ0FBQ0MsSUFGZixDQUFyQjtBQUlBLE1BQU1FLE1BQU0sR0FBR0QsZ0VBQVEsQ0FDckJMLGlCQUFpQixDQUFDTSxNQURHLEVBRXJCSCxpQ0FBaUMsQ0FBQ0csTUFGYixDQUF2QjtBQUlBLE1BQU1DLFNBQVMsR0FBR1AsaUJBQWlCLENBQUNPLFNBQXBDO0FBQ0EsTUFBTUMsV0FBVyxHQUFHSixJQUFJLElBQUlBLElBQUksQ0FBQ0ssSUFBTCxDQUFVQyxJQUFWLENBQWUsVUFBQ0MsQ0FBRCxFQUFZQyxDQUFaO0FBQUEsV0FBMEJELENBQUMsR0FBR0MsQ0FBOUI7QUFBQSxHQUFmLENBQTVCO0FBRUEsU0FDRSw0REFDRyxDQUFDTCxTQUFELElBQWNELE1BQU0sQ0FBQ08sTUFBUCxHQUFnQixDQUE5QixHQUNDUCxNQUFNLENBQUNRLEdBQVAsQ0FBVyxVQUFDQyxLQUFEO0FBQUEsV0FDVCxNQUFDLCtFQUFEO0FBQWEsU0FBRyxFQUFFQSxLQUFsQjtBQUF5QixhQUFPLEVBQUVBLEtBQWxDO0FBQXlDLFVBQUksRUFBQyxPQUE5QztBQUFzRCxjQUFRLE1BQTlEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFEUztBQUFBLEdBQVgsQ0FERCxHQUlHUixTQUFTLEdBQ1gsTUFBQyxrRkFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywyRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FEVyxHQUtYLE1BQUMscUZBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUlFLE1BQUMsb0ZBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSx1QkFKRixFQUtHQSxTQUFTLEdBQ1IsTUFBQyxrRkFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywyRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FEUSxHQUlOQyxXQUFXLElBQ2JBLFdBQVcsQ0FBQ0ssTUFBWixLQUF1QixDQURyQixJQUVGLENBQUNOLFNBRkMsSUFHRkQsTUFBTSxDQUFDTyxNQUFQLEtBQWtCLENBSGhCLEdBSUYsTUFBQyxnRkFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBSkUsR0FNRkwsV0FBVyxJQUNULE1BQUMsOERBQUQ7QUFDRSxlQUFXLEVBQUVBLFdBRGY7QUFFRSxRQUFJLEVBQUVRLCtEQUFnQixDQUFDQyxJQUZ6QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBaEJOLENBVkosQ0FERjtBQXFDRCxDQS9ETTs7R0FBTXJCLFU7VUFHZUssNEQsRUFNZ0JBLDRELEVBTTdCSSx3RCxFQUlFQSx3RDs7O0tBbkJKVCxVIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmRlZjBmNTFhZmNkZTNmNGNkZTQwLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XHJcblxyXG5pbXBvcnQgeyB1c2VSZXF1ZXN0IH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlUmVxdWVzdCc7XHJcbmltcG9ydCB7IGdldF90aGVfbGF0ZXN0X3J1bnMgfSBmcm9tICcuLi8uLi9jb25maWcvY29uZmlnJztcclxuaW1wb3J0IHtcclxuICBTcGlubmVyV3JhcHBlcixcclxuICBTcGlubmVyLFxyXG4gIExhdGVzdFJ1bnNUdGl0bGUsXHJcbiAgTGF0ZXN0UnVuc1NlY3Rpb24sXHJcbiAgU3R5bGVkQWxlcnQsXHJcbn0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9zZWFyY2gvc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCB7IE5vUmVzdWx0c0ZvdW5kIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9zZWFyY2gvbm9SZXN1bHRzRm91bmQnO1xyXG5pbXBvcnQgeyBzdG9yZSB9IGZyb20gJy4uLy4uL2NvbnRleHRzL2xlZnRTaWRlQ29udGV4dCc7XHJcbmltcG9ydCB7IHVzZU5ld2VyIH0gZnJvbSAnLi4vLi4vaG9va3MvdXNlTmV3ZXInO1xyXG5pbXBvcnQgeyBmdW5jdGlvbnNfY29uZmlnIH0gZnJvbSAnLi4vLi4vY29uZmlnL2NvbmZpZyc7XHJcbmltcG9ydCB7IExpdmVNb2RlQnV0dG9uIH0gZnJvbSAnLi4vbGl2ZU1vZGVCdXR0b24nO1xyXG5pbXBvcnQgeyBDdXN0b21EaXYgfSBmcm9tICcuLi9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHsgTGF0ZXN0UnVuc0xpc3QgfSBmcm9tICcuL2xhdGVzdFJ1bnNMaXN0JztcclxuXHJcbmV4cG9ydCBjb25zdCBMYXRlc3RSdW5zID0gKCkgPT4ge1xyXG4gIGNvbnN0IHsgdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiB9ID0gUmVhY3QudXNlQ29udGV4dChzdG9yZSk7XHJcblxyXG4gIGNvbnN0IGRhdGFfZ2V0X2J5X21vdW50ID0gdXNlUmVxdWVzdChcclxuICAgIGdldF90aGVfbGF0ZXN0X3J1bnModXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiksXHJcbiAgICB7fSxcclxuICAgIFtdXHJcbiAgKTtcclxuXHJcbiAgY29uc3QgZGF0YV9nZXRfYnlfbm90X29sZGVyX3RoYW5fdXBkYXRlID0gdXNlUmVxdWVzdChcclxuICAgIGdldF90aGVfbGF0ZXN0X3J1bnModXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbiksXHJcbiAgICB7fSxcclxuICAgIFt1cGRhdGVkX2J5X25vdF9vbGRlcl90aGFuXVxyXG4gICk7XHJcblxyXG4gIGNvbnN0IGRhdGEgPSB1c2VOZXdlcihcclxuICAgIGRhdGFfZ2V0X2J5X21vdW50LmRhdGEsXHJcbiAgICBkYXRhX2dldF9ieV9ub3Rfb2xkZXJfdGhhbl91cGRhdGUuZGF0YVxyXG4gICk7XHJcbiAgY29uc3QgZXJyb3JzID0gdXNlTmV3ZXIoXHJcbiAgICBkYXRhX2dldF9ieV9tb3VudC5lcnJvcnMsXHJcbiAgICBkYXRhX2dldF9ieV9ub3Rfb2xkZXJfdGhhbl91cGRhdGUuZXJyb3JzXHJcbiAgKTtcclxuICBjb25zdCBpc0xvYWRpbmcgPSBkYXRhX2dldF9ieV9tb3VudC5pc0xvYWRpbmc7XHJcbiAgY29uc3QgbGF0ZXN0X3J1bnMgPSBkYXRhICYmIGRhdGEucnVucy5zb3J0KChhOiBudW1iZXIsIGI6IG51bWJlcikgPT4gYSAtIGIpO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPD5cclxuICAgICAgeyFpc0xvYWRpbmcgJiYgZXJyb3JzLmxlbmd0aCA+IDAgPyAoXHJcbiAgICAgICAgZXJyb3JzLm1hcCgoZXJyb3I6IHN0cmluZykgPT4gKFxyXG4gICAgICAgICAgPFN0eWxlZEFsZXJ0IGtleT17ZXJyb3J9IG1lc3NhZ2U9e2Vycm9yfSB0eXBlPVwiZXJyb3JcIiBzaG93SWNvbiAvPlxyXG4gICAgICAgICkpXHJcbiAgICAgICkgOiBpc0xvYWRpbmcgPyAoXHJcbiAgICAgICAgPFNwaW5uZXJXcmFwcGVyPlxyXG4gICAgICAgICAgPFNwaW5uZXIgLz5cclxuICAgICAgICA8L1NwaW5uZXJXcmFwcGVyPlxyXG4gICAgICApIDogKFxyXG4gICAgICAgIDxMYXRlc3RSdW5zU2VjdGlvbj5cclxuICAgICAgICAgIHsvKiA8Q3VzdG9tRGl2IGRpc3BsYXk9XCJmbGV4XCIganVzdGlmeWNvbnRlbnQ9XCJmbGV4LWVuZFwiIHdpZHRoPVwiYXV0b1wiPlxyXG4gICAgICAgICAgICA8TGl2ZU1vZGVCdXR0b24gLz5cclxuICAgICAgICAgIDwvQ3VzdG9tRGl2PiAqL31cclxuICAgICAgICAgIDxMYXRlc3RSdW5zVHRpdGxlPlRoZSBsYXRlc3QgcnVuczwvTGF0ZXN0UnVuc1R0aXRsZT5cclxuICAgICAgICAgIHtpc0xvYWRpbmcgPyAoXHJcbiAgICAgICAgICAgIDxTcGlubmVyV3JhcHBlcj5cclxuICAgICAgICAgICAgICA8U3Bpbm5lciAvPlxyXG4gICAgICAgICAgICA8L1NwaW5uZXJXcmFwcGVyPlxyXG4gICAgICAgICAgKSA6IGxhdGVzdF9ydW5zICYmXHJcbiAgICAgICAgICAgIGxhdGVzdF9ydW5zLmxlbmd0aCA9PT0gMCAmJlxyXG4gICAgICAgICAgICAhaXNMb2FkaW5nICYmXHJcbiAgICAgICAgICAgIGVycm9ycy5sZW5ndGggPT09IDAgPyAoXHJcbiAgICAgICAgICAgIDxOb1Jlc3VsdHNGb3VuZCAvPlxyXG4gICAgICAgICAgKSA6IChcclxuICAgICAgICAgICAgbGF0ZXN0X3J1bnMgJiYgKFxyXG4gICAgICAgICAgICAgIDxMYXRlc3RSdW5zTGlzdFxyXG4gICAgICAgICAgICAgICAgbGF0ZXN0X3J1bnM9e2xhdGVzdF9ydW5zfVxyXG4gICAgICAgICAgICAgICAgbW9kZT17ZnVuY3Rpb25zX2NvbmZpZy5tb2RlfVxyXG4gICAgICAgICAgICAgIC8+XHJcbiAgICAgICAgICAgIClcclxuICAgICAgICAgICl9XHJcbiAgICAgICAgPC9MYXRlc3RSdW5zU2VjdGlvbj5cclxuICAgICAgKX1cclxuICAgIDwvPlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=