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
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _ant_design_icons__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! @ant-design/icons */ "./node_modules/@ant-design/icons/es/index.js");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../styledComponents */ "./components/styledComponents.ts");
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/plots/zoomedPlots/menu.tsx",
    _this = undefined;

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];




// const open_a_new_tab = (query: string) => {
//   consol
//   // const current_root = window.location.href.split('/?')[0];
//   // open_a_new_tab(`${current_root}/?${query}`);
//   // window.open(query, '_blank');
// };
console.log('ss');
var ZoomedPlotMenu = function ZoomedPlotMenu(_ref) {
  var options = _ref.options;

  var plotMenu = function plotMenu(options) {
    return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 22,
        columnNumber: 5
      }
    }, options.map(function (option) {
      if (option.value === 'overlay') {
        return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"].Item, {
          key: option.value,
          onClick: function onClick() {
            var query = option.action ? option.action() : ''; // open_a_new_tab(query as string)
          },
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 26,
            columnNumber: 13
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          display: "flex",
          justifycontent: "space-around",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 32,
            columnNumber: 15
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 33,
            columnNumber: 17
          }
        }, option.icon), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 34,
            columnNumber: 17
          }
        }, option.label)));
      } else {
        return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"].Item, {
          key: option.value,
          onClick: function onClick() {
            option.action && option.action(option.value);
          },
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 40,
            columnNumber: 13
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          display: "flex",
          justifycontent: "space-around",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 46,
            columnNumber: 15
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 47,
            columnNumber: 17
          }
        }, option.icon), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 48,
            columnNumber: 17
          }
        }, option.label)));
      }
    }));
  };

  return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Row"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 58,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 59,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Dropdown"], {
    overlay: plotMenu(options),
    trigger: ['hover'],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 60,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 61,
      columnNumber: 11
    }
  }, "More ", __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["DownOutlined"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 62,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy9tZW51LnRzeCJdLCJuYW1lcyI6WyJjb25zb2xlIiwibG9nIiwiWm9vbWVkUGxvdE1lbnUiLCJvcHRpb25zIiwicGxvdE1lbnUiLCJtYXAiLCJvcHRpb24iLCJ2YWx1ZSIsInF1ZXJ5IiwiYWN0aW9uIiwiaWNvbiIsImxhYmVsIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFHQTtBQU1BO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFDQTtBQUVBQSxPQUFPLENBQUNDLEdBQVIsQ0FBYSxJQUFiO0FBQ08sSUFBTUMsY0FBYyxHQUFHLFNBQWpCQSxjQUFpQixPQUE0QjtBQUFBLE1BQXpCQyxPQUF5QixRQUF6QkEsT0FBeUI7O0FBQ3hELE1BQU1DLFFBQVEsR0FBRyxTQUFYQSxRQUFXLENBQUNELE9BQUQ7QUFBQSxXQUNmLE1BQUMseUNBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUNHQSxPQUFPLENBQUNFLEdBQVIsQ0FBWSxVQUFDQyxNQUFELEVBQXlCO0FBQ3BDLFVBQUlBLE1BQU0sQ0FBQ0MsS0FBUCxLQUFpQixTQUFyQixFQUFnQztBQUM5QixlQUNFLE1BQUMseUNBQUQsQ0FBTSxJQUFOO0FBQ0UsYUFBRyxFQUFFRCxNQUFNLENBQUNDLEtBRGQ7QUFFRSxpQkFBTyxFQUFFLG1CQUFNO0FBQ2IsZ0JBQU1DLEtBQUssR0FBR0YsTUFBTSxDQUFDRyxNQUFQLEdBQWdCSCxNQUFNLENBQUNHLE1BQVAsRUFBaEIsR0FBa0MsRUFBaEQsQ0FEYSxDQUViO0FBQ0QsV0FMSDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBTUUsTUFBQywyREFBRDtBQUFXLGlCQUFPLEVBQUMsTUFBbkI7QUFBMEIsd0JBQWMsRUFBQyxjQUF6QztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQ0UsTUFBQywyREFBRDtBQUFXLGVBQUssRUFBQyxHQUFqQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQXNCSCxNQUFNLENBQUNJLElBQTdCLENBREYsRUFFRSxNQUFDLDJEQUFEO0FBQVcsZUFBSyxFQUFDLEdBQWpCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FBc0JKLE1BQU0sQ0FBQ0ssS0FBN0IsQ0FGRixDQU5GLENBREY7QUFhRCxPQWRELE1BY087QUFDTCxlQUNFLE1BQUMseUNBQUQsQ0FBTSxJQUFOO0FBQ0UsYUFBRyxFQUFFTCxNQUFNLENBQUNDLEtBRGQ7QUFFRSxpQkFBTyxFQUFFLG1CQUFNO0FBQ2JELGtCQUFNLENBQUNHLE1BQVAsSUFBaUJILE1BQU0sQ0FBQ0csTUFBUCxDQUFjSCxNQUFNLENBQUNDLEtBQXJCLENBQWpCO0FBQ0QsV0FKSDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBTUUsTUFBQywyREFBRDtBQUFXLGlCQUFPLEVBQUMsTUFBbkI7QUFBMEIsd0JBQWMsRUFBQyxjQUF6QztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQ0UsTUFBQywyREFBRDtBQUFXLGVBQUssRUFBQyxHQUFqQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQXNCRCxNQUFNLENBQUNJLElBQTdCLENBREYsRUFFRSxNQUFDLDJEQUFEO0FBQVcsZUFBSyxFQUFDLEdBQWpCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FBc0JKLE1BQU0sQ0FBQ0ssS0FBN0IsQ0FGRixDQU5GLENBREY7QUFhRDtBQUNGLEtBOUJBLENBREgsQ0FEZTtBQUFBLEdBQWpCOztBQW9DQSxTQUNFLE1BQUMsd0NBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsd0NBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsNkNBQUQ7QUFBVSxXQUFPLEVBQUVQLFFBQVEsQ0FBQ0QsT0FBRCxDQUEzQjtBQUFzQyxXQUFPLEVBQUUsQ0FBQyxPQUFELENBQS9DO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDJDQUFEO0FBQVEsUUFBSSxFQUFDLE1BQWI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxjQUNPLE1BQUMsOERBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURQLENBREYsQ0FERixDQURGLENBREY7QUFXRCxDQWhETTtLQUFNRCxjIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmM4YmUyYjBhMjAxOGY3NjQzYWJhLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IE1lbnUsIERyb3Bkb3duLCBSb3csIENvbCwgQnV0dG9uIH0gZnJvbSAnYW50ZCc7XHJcbmltcG9ydCB7IERvd25PdXRsaW5lZCB9IGZyb20gJ0BhbnQtZGVzaWduL2ljb25zJztcclxuXHJcbmltcG9ydCB7IE9wdGlvblByb3BzIH0gZnJvbSAnLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyBDdXN0b21EaXYgfSBmcm9tICcuLi8uLi9zdHlsZWRDb21wb25lbnRzJztcclxuXHJcbmV4cG9ydCBpbnRlcmZhY2UgTWVudVByb3BzIHtcclxuICBvcHRpb25zOiBPcHRpb25Qcm9wc1tdO1xyXG59XHJcblxyXG4vLyBjb25zdCBvcGVuX2FfbmV3X3RhYiA9IChxdWVyeTogc3RyaW5nKSA9PiB7XHJcbi8vICAgY29uc29sXHJcbi8vICAgLy8gY29uc3QgY3VycmVudF9yb290ID0gd2luZG93LmxvY2F0aW9uLmhyZWYuc3BsaXQoJy8/JylbMF07XHJcbi8vICAgLy8gb3Blbl9hX25ld190YWIoYCR7Y3VycmVudF9yb290fS8/JHtxdWVyeX1gKTtcclxuLy8gICAvLyB3aW5kb3cub3BlbihxdWVyeSwgJ19ibGFuaycpO1xyXG4vLyB9O1xyXG5cclxuY29uc29sZS5sb2coICdzcycpXHJcbmV4cG9ydCBjb25zdCBab29tZWRQbG90TWVudSA9ICh7IG9wdGlvbnMgfTogTWVudVByb3BzKSA9PiB7XHJcbiAgY29uc3QgcGxvdE1lbnUgPSAob3B0aW9uczogT3B0aW9uUHJvcHNbXSkgPT4gKFxyXG4gICAgPE1lbnU+XHJcbiAgICAgIHtvcHRpb25zLm1hcCgob3B0aW9uOiBPcHRpb25Qcm9wcykgPT4ge1xyXG4gICAgICAgIGlmIChvcHRpb24udmFsdWUgPT09ICdvdmVybGF5Jykge1xyXG4gICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgPE1lbnUuSXRlbVxyXG4gICAgICAgICAgICAgIGtleT17b3B0aW9uLnZhbHVlfVxyXG4gICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICAgIGNvbnN0IHF1ZXJ5ID0gb3B0aW9uLmFjdGlvbiA/IG9wdGlvbi5hY3Rpb24oKSA6ICcnO1xyXG4gICAgICAgICAgICAgICAgLy8gb3Blbl9hX25ld190YWIocXVlcnkgYXMgc3RyaW5nKVxyXG4gICAgICAgICAgICAgIH19PlxyXG4gICAgICAgICAgICAgIDxDdXN0b21EaXYgZGlzcGxheT1cImZsZXhcIiBqdXN0aWZ5Y29udGVudD1cInNwYWNlLWFyb3VuZFwiPlxyXG4gICAgICAgICAgICAgICAgPEN1c3RvbURpdiBzcGFjZT1cIjJcIj57b3B0aW9uLmljb259PC9DdXN0b21EaXY+XHJcbiAgICAgICAgICAgICAgICA8Q3VzdG9tRGl2IHNwYWNlPVwiMlwiPntvcHRpb24ubGFiZWx9PC9DdXN0b21EaXY+XHJcbiAgICAgICAgICAgICAgPC9DdXN0b21EaXY+XHJcbiAgICAgICAgICAgIDwvTWVudS5JdGVtPlxyXG4gICAgICAgICAgKVxyXG4gICAgICAgIH0gZWxzZSB7XHJcbiAgICAgICAgICByZXR1cm4gKFxyXG4gICAgICAgICAgICA8TWVudS5JdGVtXHJcbiAgICAgICAgICAgICAga2V5PXtvcHRpb24udmFsdWV9XHJcbiAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgICAgb3B0aW9uLmFjdGlvbiAmJiBvcHRpb24uYWN0aW9uKG9wdGlvbi52YWx1ZSk7XHJcbiAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgPlxyXG4gICAgICAgICAgICAgIDxDdXN0b21EaXYgZGlzcGxheT1cImZsZXhcIiBqdXN0aWZ5Y29udGVudD1cInNwYWNlLWFyb3VuZFwiPlxyXG4gICAgICAgICAgICAgICAgPEN1c3RvbURpdiBzcGFjZT1cIjJcIj57b3B0aW9uLmljb259PC9DdXN0b21EaXY+XHJcbiAgICAgICAgICAgICAgICA8Q3VzdG9tRGl2IHNwYWNlPVwiMlwiPntvcHRpb24ubGFiZWx9PC9DdXN0b21EaXY+XHJcbiAgICAgICAgICAgICAgPC9DdXN0b21EaXY+XHJcbiAgICAgICAgICAgIDwvTWVudS5JdGVtPlxyXG4gICAgICAgICAgKVxyXG4gICAgICAgIH1cclxuICAgICAgfSl9XHJcbiAgICA8L01lbnU+XHJcbiAgKTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxSb3c+XHJcbiAgICAgIDxDb2w+XHJcbiAgICAgICAgPERyb3Bkb3duIG92ZXJsYXk9e3Bsb3RNZW51KG9wdGlvbnMpfSB0cmlnZ2VyPXtbJ2hvdmVyJ119PlxyXG4gICAgICAgICAgPEJ1dHRvbiB0eXBlPVwibGlua1wiPlxyXG4gICAgICAgICAgICBNb3JlIDxEb3duT3V0bGluZWQgLz5cclxuICAgICAgICAgIDwvQnV0dG9uPlxyXG4gICAgICAgIDwvRHJvcGRvd24+XHJcbiAgICAgIDwvQ29sPlxyXG4gICAgPC9Sb3c+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==