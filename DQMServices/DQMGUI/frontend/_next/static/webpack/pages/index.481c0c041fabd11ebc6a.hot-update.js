webpackHotUpdate_N_E("pages/index",{

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
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");


var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/plots/plot/plotSearch/index.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_1__["createElement"];





var PlotSearch = function PlotSearch(_ref) {
  _s();

  var isLoadingFolders = _ref.isLoadingFolders;
  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"])();

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_1__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_5__["store"]),
      set_plot_search = _React$useContext.set_plot_search;

  var query = router.query;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_1__["useState"](query.plot_search),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState, 2),
      plotName = _React$useState2[0],
      setPlotName = _React$useState2[1];

  react__WEBPACK_IMPORTED_MODULE_1__["useEffect"](function () {
    if (plotName) {
      set_plot_search(plotName);
    } // const params = getChangedQueryParams({ plot_search: plotName }, query);
    // changeRouter(params);

  }, [plotName]);
  return react__WEBPACK_IMPORTED_MODULE_1__["useMemo"](function () {
    return __jsx(antd_lib_form_Form__WEBPACK_IMPORTED_MODULE_2___default.a, {
      onChange: function onChange(e) {
        return setPlotName(e.target.value);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 35,
        columnNumber: 7
      }
    }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 36,
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
        lineNumber: 37,
        columnNumber: 11
      }
    })));
  }, [plotName]);
};

_s(PlotSearch, "3sBn6jcc8kgdyEcLkroVA1ovY58=", false, function () {
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy9wbG90L3Bsb3RTZWFyY2gvaW5kZXgudHN4Il0sIm5hbWVzIjpbIlBsb3RTZWFyY2giLCJpc0xvYWRpbmdGb2xkZXJzIiwicm91dGVyIiwidXNlUm91dGVyIiwiUmVhY3QiLCJzdG9yZSIsInNldF9wbG90X3NlYXJjaCIsInF1ZXJ5IiwicGxvdF9zZWFyY2giLCJwbG90TmFtZSIsInNldFBsb3ROYW1lIiwiZSIsInRhcmdldCIsInZhbHVlIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUVBO0FBQ0E7QUFNQTtBQU1PLElBQU1BLFVBQVUsR0FBRyxTQUFiQSxVQUFhLE9BQTJDO0FBQUE7O0FBQUEsTUFBeENDLGdCQUF3QyxRQUF4Q0EsZ0JBQXdDO0FBQ25FLE1BQU1DLE1BQU0sR0FBR0MsNkRBQVMsRUFBeEI7O0FBRG1FLDBCQUV6Q0MsZ0RBQUEsQ0FBaUJDLCtEQUFqQixDQUZ5QztBQUFBLE1BRTVEQyxlQUY0RCxxQkFFNURBLGVBRjREOztBQUduRSxNQUFNQyxLQUFpQixHQUFHTCxNQUFNLENBQUNLLEtBQWpDOztBQUhtRSx3QkFJbkNILDhDQUFBLENBQzlCRyxLQUFLLENBQUNDLFdBRHdCLENBSm1DO0FBQUE7QUFBQSxNQUk1REMsUUFKNEQ7QUFBQSxNQUlsREMsV0FKa0Q7O0FBUW5FTixpREFBQSxDQUFnQixZQUFNO0FBQ3BCLFFBQUdLLFFBQUgsRUFBWTtBQUNWSCxxQkFBZSxDQUFDRyxRQUFELENBQWY7QUFDRCxLQUhtQixDQUlwQjtBQUNBOztBQUNELEdBTkQsRUFNRyxDQUFDQSxRQUFELENBTkg7QUFRQSxTQUFPTCw2Q0FBQSxDQUFjLFlBQU07QUFDekIsV0FDRSxNQUFDLHlEQUFEO0FBQU0sY0FBUSxFQUFFLGtCQUFDTyxDQUFEO0FBQUEsZUFBWUQsV0FBVyxDQUFDQyxDQUFDLENBQUNDLE1BQUYsQ0FBU0MsS0FBVixDQUF2QjtBQUFBLE9BQWhCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FDRSxNQUFDLGdFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FDRSxNQUFDLDhEQUFEO0FBQ0Usa0JBQVksRUFBRU4sS0FBSyxDQUFDQyxXQUR0QjtBQUVFLGFBQU8sRUFBRVAsZ0JBRlg7QUFHRSxRQUFFLEVBQUMsYUFITDtBQUlFLGlCQUFXLEVBQUMsaUJBSmQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxNQURGLENBREYsQ0FERjtBQVlELEdBYk0sRUFhSixDQUFDUSxRQUFELENBYkksQ0FBUDtBQWNELENBOUJNOztHQUFNVCxVO1VBQ0lHLHFEOzs7S0FESkgsVSIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC40ODFjMGMwNDFmYWJkMTFlYmM2YS5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xuaW1wb3J0IEZvcm0gZnJvbSAnYW50ZC9saWIvZm9ybS9Gb3JtJztcblxuaW1wb3J0IHsgdXNlUm91dGVyIH0gZnJvbSAnbmV4dC9yb3V0ZXInO1xuaW1wb3J0IHsgU3R5bGVkRm9ybUl0ZW0sIFN0eWxlZFNlYXJjaCB9IGZyb20gJy4uLy4uLy4uL3N0eWxlZENvbXBvbmVudHMnO1xuaW1wb3J0IHsgUXVlcnlQcm9wcyB9IGZyb20gJy4uLy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcbmltcG9ydCB7XG4gIGdldENoYW5nZWRRdWVyeVBhcmFtcyxcbiAgY2hhbmdlUm91dGVyLFxufSBmcm9tICcuLi8uLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvdXRpbHMnO1xuaW1wb3J0IHsgc3RvcmUgfSBmcm9tICcuLi8uLi8uLi8uLi9jb250ZXh0cy9sZWZ0U2lkZUNvbnRleHQnO1xuXG5pbnRlcmZhY2UgUGxvdFNlYXJjaFByb3BzIHtcbiAgaXNMb2FkaW5nRm9sZGVyczogYm9vbGVhbjtcbn1cblxuZXhwb3J0IGNvbnN0IFBsb3RTZWFyY2ggPSAoeyBpc0xvYWRpbmdGb2xkZXJzIH06IFBsb3RTZWFyY2hQcm9wcykgPT4ge1xuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcbiAgY29uc3Qge3NldF9wbG90X3NlYXJjaH0gPSBSZWFjdC51c2VDb250ZXh0KHN0b3JlKVxuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcbiAgY29uc3QgW3Bsb3ROYW1lLCBzZXRQbG90TmFtZV0gPSBSZWFjdC51c2VTdGF0ZTxzdHJpbmcgfCB1bmRlZmluZWQ+KFxuICAgIHF1ZXJ5LnBsb3Rfc2VhcmNoXG4gICk7XG5cbiAgUmVhY3QudXNlRWZmZWN0KCgpID0+IHtcbiAgICBpZihwbG90TmFtZSl7XG4gICAgICBzZXRfcGxvdF9zZWFyY2gocGxvdE5hbWUpXG4gICAgfVxuICAgIC8vIGNvbnN0IHBhcmFtcyA9IGdldENoYW5nZWRRdWVyeVBhcmFtcyh7IHBsb3Rfc2VhcmNoOiBwbG90TmFtZSB9LCBxdWVyeSk7XG4gICAgLy8gY2hhbmdlUm91dGVyKHBhcmFtcyk7XG4gIH0sIFtwbG90TmFtZV0pO1xuXG4gIHJldHVybiBSZWFjdC51c2VNZW1vKCgpID0+IHtcbiAgICByZXR1cm4gKFxuICAgICAgPEZvcm0gb25DaGFuZ2U9eyhlOiBhbnkpID0+IHNldFBsb3ROYW1lKGUudGFyZ2V0LnZhbHVlKX0+XG4gICAgICAgIDxTdHlsZWRGb3JtSXRlbT5cbiAgICAgICAgICA8U3R5bGVkU2VhcmNoXG4gICAgICAgICAgICBkZWZhdWx0VmFsdWU9e3F1ZXJ5LnBsb3Rfc2VhcmNofVxuICAgICAgICAgICAgbG9hZGluZz17aXNMb2FkaW5nRm9sZGVyc31cbiAgICAgICAgICAgIGlkPVwicGxvdF9zZWFyY2hcIlxuICAgICAgICAgICAgcGxhY2Vob2xkZXI9XCJFbnRlciBwbG90IG5hbWVcIlxuICAgICAgICAgIC8+XG4gICAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XG4gICAgICA8L0Zvcm0+XG4gICAgKTtcbiAgfSwgW3Bsb3ROYW1lXSk7XG59O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==