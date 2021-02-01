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





var open_a_new_tab = function open_a_new_tab(query) {
  var current_root = window.location.href.split('/?')[0];
  open_a_new_tab("".concat(current_root, "/?").concat(query));
  window.open(query, '_blank');
};

var ZoomedPlotMenu = function ZoomedPlotMenu(_ref) {
  var options = _ref.options;

  var plotMenu = function plotMenu(options) {
    return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 20,
        columnNumber: 5
      }
    }, options.map(function (option) {
      if (option.value === 'overlay') {
        return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"].Item, {
          key: option.value,
          onClick: function onClick() {
            var query = option.action ? option.action() : '';
            open_a_new_tab(query);
          },
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 24,
            columnNumber: 13
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          display: "flex",
          justifycontent: "space-around",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 30,
            columnNumber: 15
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 31,
            columnNumber: 17
          }
        }, option.icon), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 32,
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
            lineNumber: 38,
            columnNumber: 13
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          display: "flex",
          justifycontent: "space-around",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 44,
            columnNumber: 15
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 45,
            columnNumber: 17
          }
        }, option.icon), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 46,
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
      lineNumber: 56,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 57,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Dropdown"], {
    overlay: plotMenu(options),
    trigger: ['hover'],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 58,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 59,
      columnNumber: 11
    }
  }, "More ", __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["DownOutlined"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 60,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy9tZW51LnRzeCJdLCJuYW1lcyI6WyJvcGVuX2FfbmV3X3RhYiIsInF1ZXJ5IiwiY3VycmVudF9yb290Iiwid2luZG93IiwibG9jYXRpb24iLCJocmVmIiwic3BsaXQiLCJvcGVuIiwiWm9vbWVkUGxvdE1lbnUiLCJvcHRpb25zIiwicGxvdE1lbnUiLCJtYXAiLCJvcHRpb24iLCJ2YWx1ZSIsImFjdGlvbiIsImljb24iLCJsYWJlbCJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUNBO0FBR0E7O0FBTUEsSUFBTUEsY0FBYyxHQUFHLFNBQWpCQSxjQUFpQixDQUFDQyxLQUFELEVBQW1CO0FBQ3hDLE1BQU1DLFlBQVksR0FBR0MsTUFBTSxDQUFDQyxRQUFQLENBQWdCQyxJQUFoQixDQUFxQkMsS0FBckIsQ0FBMkIsSUFBM0IsRUFBaUMsQ0FBakMsQ0FBckI7QUFDQU4sZ0JBQWMsV0FBSUUsWUFBSixlQUFxQkQsS0FBckIsRUFBZDtBQUNBRSxRQUFNLENBQUNJLElBQVAsQ0FBWU4sS0FBWixFQUFtQixRQUFuQjtBQUNELENBSkQ7O0FBTU8sSUFBTU8sY0FBYyxHQUFHLFNBQWpCQSxjQUFpQixPQUE0QjtBQUFBLE1BQXpCQyxPQUF5QixRQUF6QkEsT0FBeUI7O0FBQ3hELE1BQU1DLFFBQVEsR0FBRyxTQUFYQSxRQUFXLENBQUNELE9BQUQ7QUFBQSxXQUNmLE1BQUMseUNBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxPQUNHQSxPQUFPLENBQUNFLEdBQVIsQ0FBWSxVQUFDQyxNQUFELEVBQXlCO0FBQ3BDLFVBQUlBLE1BQU0sQ0FBQ0MsS0FBUCxLQUFpQixTQUFyQixFQUFnQztBQUM5QixlQUNFLE1BQUMseUNBQUQsQ0FBTSxJQUFOO0FBQ0UsYUFBRyxFQUFFRCxNQUFNLENBQUNDLEtBRGQ7QUFFRSxpQkFBTyxFQUFFLG1CQUFNO0FBQ2IsZ0JBQU1aLEtBQUssR0FBR1csTUFBTSxDQUFDRSxNQUFQLEdBQWdCRixNQUFNLENBQUNFLE1BQVAsRUFBaEIsR0FBa0MsRUFBaEQ7QUFDQWQsMEJBQWMsQ0FBQ0MsS0FBRCxDQUFkO0FBQ0QsV0FMSDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBTUUsTUFBQywyREFBRDtBQUFXLGlCQUFPLEVBQUMsTUFBbkI7QUFBMEIsd0JBQWMsRUFBQyxjQUF6QztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQ0UsTUFBQywyREFBRDtBQUFXLGVBQUssRUFBQyxHQUFqQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQXNCVyxNQUFNLENBQUNHLElBQTdCLENBREYsRUFFRSxNQUFDLDJEQUFEO0FBQVcsZUFBSyxFQUFDLEdBQWpCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FBc0JILE1BQU0sQ0FBQ0ksS0FBN0IsQ0FGRixDQU5GLENBREY7QUFhRCxPQWRELE1BY087QUFDTCxlQUNFLE1BQUMseUNBQUQsQ0FBTSxJQUFOO0FBQ0UsYUFBRyxFQUFFSixNQUFNLENBQUNDLEtBRGQ7QUFFRSxpQkFBTyxFQUFFLG1CQUFNO0FBQ2JELGtCQUFNLENBQUNFLE1BQVAsSUFBaUJGLE1BQU0sQ0FBQ0UsTUFBUCxDQUFjRixNQUFNLENBQUNDLEtBQXJCLENBQWpCO0FBQ0QsV0FKSDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBTUUsTUFBQywyREFBRDtBQUFXLGlCQUFPLEVBQUMsTUFBbkI7QUFBMEIsd0JBQWMsRUFBQyxjQUF6QztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQ0UsTUFBQywyREFBRDtBQUFXLGVBQUssRUFBQyxHQUFqQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQXNCRCxNQUFNLENBQUNHLElBQTdCLENBREYsRUFFRSxNQUFDLDJEQUFEO0FBQVcsZUFBSyxFQUFDLEdBQWpCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FBc0JILE1BQU0sQ0FBQ0ksS0FBN0IsQ0FGRixDQU5GLENBREY7QUFhRDtBQUNGLEtBOUJBLENBREgsQ0FEZTtBQUFBLEdBQWpCOztBQW9DQSxTQUNFLE1BQUMsd0NBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsd0NBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsNkNBQUQ7QUFBVSxXQUFPLEVBQUVOLFFBQVEsQ0FBQ0QsT0FBRCxDQUEzQjtBQUFzQyxXQUFPLEVBQUUsQ0FBQyxPQUFELENBQS9DO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDJDQUFEO0FBQVEsUUFBSSxFQUFDLE1BQWI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxjQUNPLE1BQUMsOERBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQURQLENBREYsQ0FERixDQURGLENBREY7QUFXRCxDQWhETTtLQUFNRCxjIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjVjYTRhYWU1YmY4NTYwNzVhZjU0LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XHJcbmltcG9ydCB7IE1lbnUsIERyb3Bkb3duLCBSb3csIENvbCwgQnV0dG9uIH0gZnJvbSAnYW50ZCc7XHJcbmltcG9ydCB7IERvd25PdXRsaW5lZCB9IGZyb20gJ0BhbnQtZGVzaWduL2ljb25zJztcclxuXHJcbmltcG9ydCB7IE9wdGlvblByb3BzIH0gZnJvbSAnLi4vLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyBDdXN0b21EaXYgfSBmcm9tICcuLi8uLi9zdHlsZWRDb21wb25lbnRzJztcclxuXHJcbmV4cG9ydCBpbnRlcmZhY2UgTWVudVByb3BzIHtcclxuICBvcHRpb25zOiBPcHRpb25Qcm9wc1tdO1xyXG59XHJcblxyXG5jb25zdCBvcGVuX2FfbmV3X3RhYiA9IChxdWVyeTogc3RyaW5nKSA9PiB7XHJcbiAgY29uc3QgY3VycmVudF9yb290ID0gd2luZG93LmxvY2F0aW9uLmhyZWYuc3BsaXQoJy8/JylbMF07XHJcbiAgb3Blbl9hX25ld190YWIoYCR7Y3VycmVudF9yb290fS8/JHtxdWVyeX1gKTtcclxuICB3aW5kb3cub3BlbihxdWVyeSwgJ19ibGFuaycpO1xyXG59O1xyXG5cclxuZXhwb3J0IGNvbnN0IFpvb21lZFBsb3RNZW51ID0gKHsgb3B0aW9ucyB9OiBNZW51UHJvcHMpID0+IHtcclxuICBjb25zdCBwbG90TWVudSA9IChvcHRpb25zOiBPcHRpb25Qcm9wc1tdKSA9PiAoXHJcbiAgICA8TWVudT5cclxuICAgICAge29wdGlvbnMubWFwKChvcHRpb246IE9wdGlvblByb3BzKSA9PiB7XHJcbiAgICAgICAgaWYgKG9wdGlvbi52YWx1ZSA9PT0gJ292ZXJsYXknKSB7XHJcbiAgICAgICAgICByZXR1cm4gKFxyXG4gICAgICAgICAgICA8TWVudS5JdGVtXHJcbiAgICAgICAgICAgICAga2V5PXtvcHRpb24udmFsdWV9XHJcbiAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgICAgY29uc3QgcXVlcnkgPSBvcHRpb24uYWN0aW9uID8gb3B0aW9uLmFjdGlvbigpIDogJyc7XHJcbiAgICAgICAgICAgICAgICBvcGVuX2FfbmV3X3RhYihxdWVyeSBhcyBzdHJpbmcpXHJcbiAgICAgICAgICAgICAgfX0+XHJcbiAgICAgICAgICAgICAgPEN1c3RvbURpdiBkaXNwbGF5PVwiZmxleFwiIGp1c3RpZnljb250ZW50PVwic3BhY2UtYXJvdW5kXCI+XHJcbiAgICAgICAgICAgICAgICA8Q3VzdG9tRGl2IHNwYWNlPVwiMlwiPntvcHRpb24uaWNvbn08L0N1c3RvbURpdj5cclxuICAgICAgICAgICAgICAgIDxDdXN0b21EaXYgc3BhY2U9XCIyXCI+e29wdGlvbi5sYWJlbH08L0N1c3RvbURpdj5cclxuICAgICAgICAgICAgICA8L0N1c3RvbURpdj5cclxuICAgICAgICAgICAgPC9NZW51Lkl0ZW0+XHJcbiAgICAgICAgICApXHJcbiAgICAgICAgfSBlbHNlIHtcclxuICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgIDxNZW51Lkl0ZW1cclxuICAgICAgICAgICAgICBrZXk9e29wdGlvbi52YWx1ZX1cclxuICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICBvcHRpb24uYWN0aW9uICYmIG9wdGlvbi5hY3Rpb24ob3B0aW9uLnZhbHVlKTtcclxuICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICA+XHJcbiAgICAgICAgICAgICAgPEN1c3RvbURpdiBkaXNwbGF5PVwiZmxleFwiIGp1c3RpZnljb250ZW50PVwic3BhY2UtYXJvdW5kXCI+XHJcbiAgICAgICAgICAgICAgICA8Q3VzdG9tRGl2IHNwYWNlPVwiMlwiPntvcHRpb24uaWNvbn08L0N1c3RvbURpdj5cclxuICAgICAgICAgICAgICAgIDxDdXN0b21EaXYgc3BhY2U9XCIyXCI+e29wdGlvbi5sYWJlbH08L0N1c3RvbURpdj5cclxuICAgICAgICAgICAgICA8L0N1c3RvbURpdj5cclxuICAgICAgICAgICAgPC9NZW51Lkl0ZW0+XHJcbiAgICAgICAgICApXHJcbiAgICAgICAgfVxyXG4gICAgICB9KX1cclxuICAgIDwvTWVudT5cclxuICApO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPFJvdz5cclxuICAgICAgPENvbD5cclxuICAgICAgICA8RHJvcGRvd24gb3ZlcmxheT17cGxvdE1lbnUob3B0aW9ucyl9IHRyaWdnZXI9e1snaG92ZXInXX0+XHJcbiAgICAgICAgICA8QnV0dG9uIHR5cGU9XCJsaW5rXCI+XHJcbiAgICAgICAgICAgIE1vcmUgPERvd25PdXRsaW5lZCAvPlxyXG4gICAgICAgICAgPC9CdXR0b24+XHJcbiAgICAgICAgPC9Ecm9wZG93bj5cclxuICAgICAgPC9Db2w+XHJcbiAgICA8L1Jvdz5cclxuICApO1xyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9