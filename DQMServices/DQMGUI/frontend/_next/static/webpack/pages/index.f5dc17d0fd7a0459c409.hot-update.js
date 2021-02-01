webpackHotUpdate_N_E("pages/index",{

/***/ "./components/plots/zoomedPlots/menu.tsx":
/*!***********************************************!*\
  !*** ./components/plots/zoomedPlots/menu.tsx ***!
  \***********************************************/
/*! exports provided: ZoomedPlotMenu */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "ZoomedPlotMenu", function() { return ZoomedPlotMenu; });
/* harmony import */ var _babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/extends */ "./node_modules/@babel/runtime/helpers/esm/extends.js");
/* harmony import */ var _babel_runtime_helpers_esm_objectWithoutProperties__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/objectWithoutProperties */ "./node_modules/@babel/runtime/helpers/esm/objectWithoutProperties.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _ant_design_icons__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! @ant-design/icons */ "./node_modules/@ant-design/icons/es/index.js");
/* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! next/link */ "./node_modules/next/link.js");
/* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_5___default = /*#__PURE__*/__webpack_require__.n(next_link__WEBPACK_IMPORTED_MODULE_5__);



var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/plots/zoomedPlots/menu.tsx",
    _this = undefined;

var __jsx = react__WEBPACK_IMPORTED_MODULE_2__["createElement"];




var ZoomedPlotMenu = function ZoomedPlotMenu(_ref) {
  var options = _ref.options,
      props = Object(_babel_runtime_helpers_esm_objectWithoutProperties__WEBPACK_IMPORTED_MODULE_1__["default"])(_ref, ["options"]);

  var plotMenu = function plotMenu(options) {
    return __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Menu"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 15,
        columnNumber: 5
      }
    }, options.map(function (option) {
      if (option.value === 'overlay') {
        return __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Menu"].Item, Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({}, props, {
          icon: option.icon,
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 19,
            columnNumber: 13
          }
        }), __jsx(next_link__WEBPACK_IMPORTED_MODULE_5___default.a, {
          href: option.url,
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 21,
            columnNumber: 15
          }
        }, __jsx("a", {
          target: "_bank",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 22,
            columnNumber: 17
          }
        }, option.label)));
      } else {
        return __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Menu"].Item, Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({}, props, {
          icon: option.icon,
          key: option.value,
          onClick: function onClick() {
            option.action && option.action(option.value);
          },
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 30,
            columnNumber: 13
          }
        }), option.label);
      }
    }));
  };

  return __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Row"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 46,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 47,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Dropdown"], {
    overlay: plotMenu(options),
    trigger: ['hover'],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 48,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Button"], {
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 49,
      columnNumber: 11
    }
  }, "More ", __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_4__["DownOutlined"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 50,
      columnNumber: 18
    }
  })))));
};
_c = ZoomedPlotMenu;

var _c;

$RefreshReg$(_c, "ZoomedPlotMenu");

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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy9tZW51LnRzeCJdLCJuYW1lcyI6WyJab29tZWRQbG90TWVudSIsIm9wdGlvbnMiLCJwcm9wcyIsInBsb3RNZW51IiwibWFwIiwib3B0aW9uIiwidmFsdWUiLCJpY29uIiwidXJsIiwibGFiZWwiLCJhY3Rpb24iXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUlBO0FBTU8sSUFBTUEsY0FBYyxHQUFHLFNBQWpCQSxjQUFpQixPQUFzQztBQUFBLE1BQW5DQyxPQUFtQyxRQUFuQ0EsT0FBbUM7QUFBQSxNQUF2QkMsS0FBdUI7O0FBQ2xFLE1BQU1DLFFBQVEsR0FBRyxTQUFYQSxRQUFXLENBQUNGLE9BQUQ7QUFBQSxXQUNmLE1BQUMseUNBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUNHQSxPQUFPLENBQUNHLEdBQVIsQ0FBWSxVQUFDQyxNQUFELEVBQXlCO0FBQ3BDLFVBQUlBLE1BQU0sQ0FBQ0MsS0FBUCxLQUFpQixTQUFyQixFQUFnQztBQUM5QixlQUNFLE1BQUMseUNBQUQsQ0FBTSxJQUFOLHlGQUFlSixLQUFmO0FBQ0UsY0FBSSxFQUFFRyxNQUFNLENBQUNFLElBRGY7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxZQUVFLE1BQUMsZ0RBQUQ7QUFBTSxjQUFJLEVBQUVGLE1BQU0sQ0FBQ0csR0FBbkI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQUNFO0FBQUcsZ0JBQU0sRUFBQyxPQUFWO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FDR0gsTUFBTSxDQUFDSSxLQURWLENBREYsQ0FGRixDQURGO0FBVUQsT0FYRCxNQVdPO0FBQ0wsZUFDRSxNQUFDLHlDQUFELENBQU0sSUFBTix5RkFDTVAsS0FETjtBQUVFLGNBQUksRUFBRUcsTUFBTSxDQUFDRSxJQUZmO0FBR0UsYUFBRyxFQUFFRixNQUFNLENBQUNDLEtBSGQ7QUFJRSxpQkFBTyxFQUFFLG1CQUFNO0FBQ2JELGtCQUFNLENBQUNLLE1BQVAsSUFBaUJMLE1BQU0sQ0FBQ0ssTUFBUCxDQUFjTCxNQUFNLENBQUNDLEtBQXJCLENBQWpCO0FBQ0QsV0FOSDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFlBT0VELE1BQU0sQ0FBQ0ksS0FQVCxDQURGO0FBV0Q7QUFDRixLQXpCQSxDQURILENBRGU7QUFBQSxHQUFqQjs7QUErQkEsU0FDRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDZDQUFEO0FBQVUsV0FBTyxFQUFFTixRQUFRLENBQUNGLE9BQUQsQ0FBM0I7QUFBc0MsV0FBTyxFQUFFLENBQUMsT0FBRCxDQUEvQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywyQ0FBRDtBQUFRLFFBQUksRUFBQyxNQUFiO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsY0FDTyxNQUFDLDhEQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFEUCxDQURGLENBREYsQ0FERixDQURGO0FBV0QsQ0EzQ007S0FBTUQsYyIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC5mNWRjMTdkMGZkN2EwNDU5YzQwOS5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xyXG5pbXBvcnQgeyBNZW51LCBEcm9wZG93biwgUm93LCBDb2wsIEJ1dHRvbiB9IGZyb20gJ2FudGQnO1xyXG5pbXBvcnQgeyBEb3duT3V0bGluZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XHJcblxyXG5pbXBvcnQgeyBPcHRpb25Qcm9wcyB9IGZyb20gJy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHsgQ3VzdG9tRGl2IH0gZnJvbSAnLi4vLi4vc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCBMaW5rIGZyb20gJ25leHQvbGluayc7XHJcblxyXG5leHBvcnQgaW50ZXJmYWNlIE1lbnVQcm9wcyB7XHJcbiAgb3B0aW9uczogT3B0aW9uUHJvcHNbXTtcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IFpvb21lZFBsb3RNZW51ID0gKHsgb3B0aW9ucywgLi4ucHJvcHMgfTogTWVudVByb3BzKSA9PiB7XHJcbiAgY29uc3QgcGxvdE1lbnUgPSAob3B0aW9uczogT3B0aW9uUHJvcHNbXSkgPT4gKFxyXG4gICAgPE1lbnU+XHJcbiAgICAgIHtvcHRpb25zLm1hcCgob3B0aW9uOiBPcHRpb25Qcm9wcykgPT4ge1xyXG4gICAgICAgIGlmIChvcHRpb24udmFsdWUgPT09ICdvdmVybGF5Jykge1xyXG4gICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgPE1lbnUuSXRlbSB7Li4ucHJvcHN9XHJcbiAgICAgICAgICAgICAgaWNvbj17b3B0aW9uLmljb259PlxyXG4gICAgICAgICAgICAgIDxMaW5rIGhyZWY9e29wdGlvbi51cmwgYXMgc3RyaW5nfT5cclxuICAgICAgICAgICAgICAgIDxhIHRhcmdldD1cIl9iYW5rXCI+XHJcbiAgICAgICAgICAgICAgICAgIHtvcHRpb24ubGFiZWx9XHJcbiAgICAgICAgICAgICAgICA8L2E+XHJcbiAgICAgICAgICAgICAgPC9MaW5rPlxyXG4gICAgICAgICAgICA8L01lbnUuSXRlbT5cclxuICAgICAgICAgIClcclxuICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgPE1lbnUuSXRlbVxyXG4gICAgICAgICAgICAgIHsuLi5wcm9wc31cclxuICAgICAgICAgICAgICBpY29uPXtvcHRpb24uaWNvbn1cclxuICAgICAgICAgICAgICBrZXk9e29wdGlvbi52YWx1ZX1cclxuICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICBvcHRpb24uYWN0aW9uICYmIG9wdGlvbi5hY3Rpb24ob3B0aW9uLnZhbHVlKTtcclxuICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICA+e29wdGlvbi5sYWJlbH1cclxuICAgICAgICAgICAgPC9NZW51Lkl0ZW0+XHJcbiAgICAgICAgICApXHJcbiAgICAgICAgfVxyXG4gICAgICB9KX1cclxuICAgIDwvTWVudSA+XHJcbiAgKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxSb3c+XHJcbiAgICAgIDxDb2w+XHJcbiAgICAgICAgPERyb3Bkb3duIG92ZXJsYXk9e3Bsb3RNZW51KG9wdGlvbnMpfSB0cmlnZ2VyPXtbJ2hvdmVyJ119PlxyXG4gICAgICAgICAgPEJ1dHRvbiB0eXBlPVwibGlua1wiPlxyXG4gICAgICAgICAgICBNb3JlIDxEb3duT3V0bGluZWQgLz5cclxuICAgICAgICAgIDwvQnV0dG9uPlxyXG4gICAgICAgIDwvRHJvcGRvd24+XHJcbiAgICAgIDwvQ29sPlxyXG4gICAgPC9Sb3c+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==