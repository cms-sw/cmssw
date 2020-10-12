webpackHotUpdate_N_E("pages/index",{

/***/ "./components/navigation/composedSearch.tsx":
/*!**************************************************!*\
  !*** ./components/navigation/composedSearch.tsx ***!
  \**************************************************/
/*! exports provided: ComposedSearch */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ComposedSearch", function() { return ComposedSearch; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var _workspaces__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../workspaces */ "./components/workspaces/index.tsx");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _plots_plot_plotSearch__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../plots/plot/plotSearch */ "./components/plots/plot/plotSearch/index.tsx");
/* harmony import */ var _containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../containers/display/styledComponents */ "./containers/display/styledComponents.tsx");
/* harmony import */ var _liveModeHeader__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ./liveModeHeader */ "./components/navigation/liveModeHeader.tsx");
/* harmony import */ var _archive_mode_header__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! ./archive_mode_header */ "./components/navigation/archive_mode_header.tsx");
var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/navigation/composedSearch.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];









var ComposedSearch = function ComposedSearch() {
  _s();

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"])();
  var query = router.query;
  var set_on_live_mode = query.run_number === '0' && query.dataset_name === '/Global/Online/ALL';
  return __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["CustomRow"], {
    width: "100%",
    display: "flex",
    justifycontent: "space-between",
    alignitems: "center",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 21,
      columnNumber: 5
    }
  }, set_on_live_mode ? __jsx(_liveModeHeader__WEBPACK_IMPORTED_MODULE_7__["LiveModeHeader"], {
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 28,
      columnNumber: 9
    }
  }) : __jsx(_archive_mode_header__WEBPACK_IMPORTED_MODULE_8__["ArchiveModeHeader"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 30,
      columnNumber: 9
    }
  }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_6__["WrapperDiv"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 32,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 33,
      columnNumber: 9
    }
  }, __jsx(_workspaces__WEBPACK_IMPORTED_MODULE_3__["default"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 34,
      columnNumber: 11
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 36,
      columnNumber: 9
    }
  }, __jsx(_plots_plot_plotSearch__WEBPACK_IMPORTED_MODULE_5__["PlotSearch"], {
    isLoadingFolders: false,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 37,
      columnNumber: 11
    }
  }))));
};

_s(ComposedSearch, "fN7XvhJ+p5oE6+Xlo0NJmXpxjC8=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_2__["useRouter"]];
});

_c = ComposedSearch;

var _c;

$RefreshReg$(_c, "ComposedSearch");

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

/***/ "./components/plots/plot/plotSearch/index.tsx":
/*!****************************************************!*\
  !*** ./components/plots/plot/plotSearch/index.tsx ***!
  \****************************************************/
/*! exports provided: PlotSearch */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "PlotSearch", function() { return PlotSearch; });
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! antd/lib/form/Form */ "./node_modules/antd/lib/form/Form.js");
/* harmony import */ var antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../../styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../../../containers/display/utils */ "./containers/display/utils.ts");


var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/plots/plot/plotSearch/index.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_1__["createElement"];





var PlotSearch = function PlotSearch(_ref) {
  _s();

  var isLoadingFolders = _ref.isLoadingFolders;
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"])();
  var query = router.query;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_1__["useState"](query.plot_search),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState, 2),
      plotName = _React$useState2[0],
      setPlotName = _React$useState2[1];

  react__WEBPACK_IMPORTED_MODULE_1__["useEffect"](function () {
    if (query.plot_search !== plotName) {
      var params = Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_5__["getChangedQueryParams"])({
        plot_search: plotName
      }, query);
      Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_5__["changeRouter"])(params);
    }
  }, [plotName]);
  return react__WEBPACK_IMPORTED_MODULE_1__["useMemo"](function () {
    return __jsx(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_2___default.a, {
      onChange: function onChange(e) {
        return setPlotName(e.target.value);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 32,
        columnNumber: 7
      }
    }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 33,
        columnNumber: 9
      }
    }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledSearch"], {
      defaultValue: query.plot_search,
      loading: isLoadingFolders,
      id: "plot_search",
      placeholder: "Enter plot name",
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 34,
        columnNumber: 11
      }
    })));
  }, [plotName]);
};

_s(PlotSearch, "qUuwOtWUsWURNKw3w2PjYEO5WgU=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"]];
});

_c = PlotSearch;

var _c;

$RefreshReg$(_c, "PlotSearch");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2NvbXBvc2VkU2VhcmNoLnRzeCIsIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy9wbG90L3Bsb3RTZWFyY2gvaW5kZXgudHN4Il0sIm5hbWVzIjpbIkNvbXBvc2VkU2VhcmNoIiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJzZXRfb25fbGl2ZV9tb2RlIiwicnVuX251bWJlciIsImRhdGFzZXRfbmFtZSIsIlBsb3RTZWFyY2giLCJpc0xvYWRpbmdGb2xkZXJzIiwiUmVhY3QiLCJwbG90X3NlYXJjaCIsInBsb3ROYW1lIiwic2V0UGxvdE5hbWUiLCJwYXJhbXMiLCJnZXRDaGFuZ2VkUXVlcnlQYXJhbXMiLCJjaGFuZ2VSb3V0ZXIiLCJlIiwidGFyZ2V0IiwidmFsdWUiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFFQTtBQUNBO0FBQ0E7QUFFQTtBQUNBO0FBQ0E7QUFFTyxJQUFNQSxjQUFjLEdBQUcsU0FBakJBLGNBQWlCLEdBQU07QUFBQTs7QUFDbEMsTUFBTUMsTUFBTSxHQUFHQyw2REFBUyxFQUF4QjtBQUNBLE1BQU1DLEtBQWlCLEdBQUdGLE1BQU0sQ0FBQ0UsS0FBakM7QUFFQSxNQUFNQyxnQkFBZ0IsR0FDcEJELEtBQUssQ0FBQ0UsVUFBTixLQUFxQixHQUFyQixJQUE0QkYsS0FBSyxDQUFDRyxZQUFOLEtBQXVCLG9CQURyRDtBQUdBLFNBQ0UsTUFBQywyREFBRDtBQUNFLFNBQUssRUFBQyxNQURSO0FBRUUsV0FBTyxFQUFDLE1BRlY7QUFHRSxrQkFBYyxFQUFDLGVBSGpCO0FBSUUsY0FBVSxFQUFDLFFBSmI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQU1HRixnQkFBZ0IsR0FDZixNQUFDLDhEQUFEO0FBQWdCLFNBQUssRUFBRUQsS0FBdkI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURlLEdBR2YsTUFBQyxzRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBVEosRUFXRSxNQUFDLCtFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLG1EQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQURGLEVBSUUsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyxpRUFBRDtBQUFZLG9CQUFnQixFQUFFLEtBQTlCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQUpGLENBWEYsQ0FERjtBQXNCRCxDQTdCTTs7R0FBTUgsYztVQUNJRSxxRDs7O0tBREpGLGM7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUNaYjtBQUNBO0FBRUE7QUFDQTtBQUVBO0FBU08sSUFBTU8sVUFBVSxHQUFHLFNBQWJBLFVBQWEsT0FBMkM7QUFBQTs7QUFBQSxNQUF4Q0MsZ0JBQXdDLFFBQXhDQSxnQkFBd0M7QUFDbkUsTUFBTVAsTUFBTSxHQUFHQyw2REFBUyxFQUF4QjtBQUNBLE1BQU1DLEtBQWlCLEdBQUdGLE1BQU0sQ0FBQ0UsS0FBakM7O0FBRm1FLHdCQUduQ00sOENBQUEsQ0FDOUJOLEtBQUssQ0FBQ08sV0FEd0IsQ0FIbUM7QUFBQTtBQUFBLE1BRzVEQyxRQUg0RDtBQUFBLE1BR2xEQyxXQUhrRDs7QUFPbkVILGlEQUFBLENBQWdCLFlBQU07QUFDcEIsUUFBR04sS0FBSyxDQUFDTyxXQUFOLEtBQXNCQyxRQUF6QixFQUFrQztBQUNoQyxVQUFNRSxNQUFNLEdBQUdDLHVGQUFxQixDQUFDO0FBQUVKLG1CQUFXLEVBQUVDO0FBQWYsT0FBRCxFQUE0QlIsS0FBNUIsQ0FBcEM7QUFDQVksb0ZBQVksQ0FBQ0YsTUFBRCxDQUFaO0FBQ0Q7QUFDRixHQUxELEVBS0csQ0FBQ0YsUUFBRCxDQUxIO0FBT0EsU0FBT0YsNkNBQUEsQ0FBYyxZQUFNO0FBQ3pCLFdBQ0UsTUFBQyx5REFBRDtBQUFNLGNBQVEsRUFBRSxrQkFBQ08sQ0FBRDtBQUFBLGVBQVlKLFdBQVcsQ0FBQ0ksQ0FBQyxDQUFDQyxNQUFGLENBQVNDLEtBQVYsQ0FBdkI7QUFBQSxPQUFoQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQ0UsTUFBQyxnRUFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQ0UsTUFBQyw4REFBRDtBQUNFLGtCQUFZLEVBQUVmLEtBQUssQ0FBQ08sV0FEdEI7QUFFRSxhQUFPLEVBQUVGLGdCQUZYO0FBR0UsUUFBRSxFQUFDLGFBSEw7QUFJRSxpQkFBVyxFQUFDLGlCQUpkO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsTUFERixDQURGLENBREY7QUFZRCxHQWJNLEVBYUosQ0FBQ0csUUFBRCxDQWJJLENBQVA7QUFjRCxDQTVCTTs7R0FBTUosVTtVQUNJTCxxRDs7O0tBREpLLFUiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguOWIxMjVjNWUyOTc4M2YxNGIyMWEuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IENvbCB9IGZyb20gJ2FudGQnO1xuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xuXG5pbXBvcnQgV29ya3NwYWNlcyBmcm9tICcuLi93b3Jrc3BhY2VzJztcbmltcG9ydCB7IEN1c3RvbVJvdyB9IGZyb20gJy4uL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgUGxvdFNlYXJjaCB9IGZyb20gJy4uL3Bsb3RzL3Bsb3QvcGxvdFNlYXJjaCc7XG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHsgV3JhcHBlckRpdiB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9zdHlsZWRDb21wb25lbnRzJztcbmltcG9ydCB7IExpdmVNb2RlSGVhZGVyIH0gZnJvbSAnLi9saXZlTW9kZUhlYWRlcic7XG5pbXBvcnQgeyBBcmNoaXZlTW9kZUhlYWRlciB9IGZyb20gJy4vYXJjaGl2ZV9tb2RlX2hlYWRlcic7XG5cbmV4cG9ydCBjb25zdCBDb21wb3NlZFNlYXJjaCA9ICgpID0+IHtcbiAgY29uc3Qgcm91dGVyID0gdXNlUm91dGVyKCk7XG4gIGNvbnN0IHF1ZXJ5OiBRdWVyeVByb3BzID0gcm91dGVyLnF1ZXJ5O1xuXG4gIGNvbnN0IHNldF9vbl9saXZlX21vZGUgPVxuICAgIHF1ZXJ5LnJ1bl9udW1iZXIgPT09ICcwJyAmJiBxdWVyeS5kYXRhc2V0X25hbWUgPT09ICcvR2xvYmFsL09ubGluZS9BTEwnO1xuXG4gIHJldHVybiAoXG4gICAgPEN1c3RvbVJvd1xuICAgICAgd2lkdGg9XCIxMDAlXCJcbiAgICAgIGRpc3BsYXk9XCJmbGV4XCJcbiAgICAgIGp1c3RpZnljb250ZW50PVwic3BhY2UtYmV0d2VlblwiXG4gICAgICBhbGlnbml0ZW1zPVwiY2VudGVyXCJcbiAgICA+XG4gICAgICB7c2V0X29uX2xpdmVfbW9kZSA/IChcbiAgICAgICAgPExpdmVNb2RlSGVhZGVyIHF1ZXJ5PXtxdWVyeX0gLz5cbiAgICAgICkgOiAoXG4gICAgICAgIDxBcmNoaXZlTW9kZUhlYWRlciAvPlxuICAgICAgKX1cbiAgICAgIDxXcmFwcGVyRGl2PlxuICAgICAgICA8Q29sPlxuICAgICAgICAgIDxXb3Jrc3BhY2VzIC8+XG4gICAgICAgIDwvQ29sPlxuICAgICAgICA8Q29sPlxuICAgICAgICAgIDxQbG90U2VhcmNoIGlzTG9hZGluZ0ZvbGRlcnM9e2ZhbHNlfSAvPlxuICAgICAgICA8L0NvbD5cbiAgICAgIDwvV3JhcHBlckRpdj5cbiAgICA8L0N1c3RvbVJvdz5cbiAgKTtcbn07XG4iLCJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XG5pbXBvcnQgRm9ybSBmcm9tICdhbnRkL2xpYi9mb3JtL0Zvcm0nO1xuXG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcic7XG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSwgU3R5bGVkU2VhcmNoIH0gZnJvbSAnLi4vLi4vLi4vc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyBRdWVyeVByb3BzIH0gZnJvbSAnLi4vLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xuaW1wb3J0IHtcbiAgZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zLFxuICBjaGFuZ2VSb3V0ZXIsXG59IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS91dGlscyc7XG5cbmludGVyZmFjZSBQbG90U2VhcmNoUHJvcHMge1xuICBpc0xvYWRpbmdGb2xkZXJzOiBib29sZWFuO1xufVxuXG5leHBvcnQgY29uc3QgUGxvdFNlYXJjaCA9ICh7IGlzTG9hZGluZ0ZvbGRlcnMgfTogUGxvdFNlYXJjaFByb3BzKSA9PiB7XG4gIGNvbnN0IHJvdXRlciA9IHVzZVJvdXRlcigpO1xuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcbiAgY29uc3QgW3Bsb3ROYW1lLCBzZXRQbG90TmFtZV0gPSBSZWFjdC51c2VTdGF0ZTxzdHJpbmcgfCB1bmRlZmluZWQ+KFxuICAgIHF1ZXJ5LnBsb3Rfc2VhcmNoXG4gICk7XG5cbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcbiAgICBpZihxdWVyeS5wbG90X3NlYXJjaCAhPT0gcGxvdE5hbWUpe1xuICAgICAgY29uc3QgcGFyYW1zID0gZ2V0Q2hhbmdlZFF1ZXJ5UGFyYW1zKHsgcGxvdF9zZWFyY2g6IHBsb3ROYW1lIH0sIHF1ZXJ5KTtcbiAgICAgIGNoYW5nZVJvdXRlcihwYXJhbXMpO1xuICAgIH1cbiAgfSwgW3Bsb3ROYW1lXSk7XG5cbiAgcmV0dXJuIFJlYWN0LnVzZU1lbW8oKCkgPT4ge1xuICAgIHJldHVybiAoXG4gICAgICA8Rm9ybSBvbkNoYW5nZT17KGU6IGFueSkgPT4gc2V0UGxvdE5hbWUoZS50YXJnZXQudmFsdWUpfT5cbiAgICAgICAgPFN0eWxlZEZvcm1JdGVtPlxuICAgICAgICAgIDxTdHlsZWRTZWFyY2hcbiAgICAgICAgICAgIGRlZmF1bHRWYWx1ZT17cXVlcnkucGxvdF9zZWFyY2h9XG4gICAgICAgICAgICBsb2FkaW5nPXtpc0xvYWRpbmdGb2xkZXJzfVxuICAgICAgICAgICAgaWQ9XCJwbG90X3NlYXJjaFwiXG4gICAgICAgICAgICBwbGFjZWhvbGRlcj1cIkVudGVyIHBsb3QgbmFtZVwiXG4gICAgICAgICAgLz5cbiAgICAgICAgPC9TdHlsZWRGb3JtSXRlbT5cbiAgICAgIDwvRm9ybT5cbiAgICApO1xuICB9LCBbcGxvdE5hbWVdKTtcbn07XG4iXSwic291cmNlUm9vdCI6IiJ9