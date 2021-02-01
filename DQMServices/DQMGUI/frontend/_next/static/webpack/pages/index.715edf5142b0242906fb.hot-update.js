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




// /
console.log('ss');
var ZoomedPlotMenu = function ZoomedPlotMenu(_ref) {
  var options = _ref.options;

  var plotMenu = function plotMenu(options) {
    return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 16,
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
            lineNumber: 20,
            columnNumber: 13
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          display: "flex",
          justifycontent: "space-around",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 26,
            columnNumber: 15
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 27,
            columnNumber: 17
          }
        }, option.icon), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 28,
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
            lineNumber: 34,
            columnNumber: 13
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          display: "flex",
          justifycontent: "space-around",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 40,
            columnNumber: 15
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 41,
            columnNumber: 17
          }
        }, option.icon), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 42,
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
      lineNumber: 52,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 53,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Dropdown"], {
    overlay: plotMenu(options),
    trigger: ['hover'],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 54,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 55,
      columnNumber: 11
    }
  }, "More ", __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["DownOutlined"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 56,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy9tZW51LnRzeCJdLCJuYW1lcyI6WyJjb25zb2xlIiwibG9nIiwiWm9vbWVkUGxvdE1lbnUiLCJvcHRpb25zIiwicGxvdE1lbnUiLCJtYXAiLCJvcHRpb24iLCJ2YWx1ZSIsInF1ZXJ5IiwiYWN0aW9uIiwib3Blbl9hX25ld190YWIiLCJpY29uIiwibGFiZWwiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUdBO0FBTUE7QUFDQUEsT0FBTyxDQUFDQyxHQUFSLENBQVksSUFBWjtBQUNPLElBQU1DLGNBQWMsR0FBRyxTQUFqQkEsY0FBaUIsT0FBNEI7QUFBQSxNQUF6QkMsT0FBeUIsUUFBekJBLE9BQXlCOztBQUN4RCxNQUFNQyxRQUFRLEdBQUcsU0FBWEEsUUFBVyxDQUFDRCxPQUFEO0FBQUEsV0FDZixNQUFDLHlDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FDR0EsT0FBTyxDQUFDRSxHQUFSLENBQVksVUFBQ0MsTUFBRCxFQUF5QjtBQUNwQyxVQUFJQSxNQUFNLENBQUNDLEtBQVAsS0FBaUIsU0FBckIsRUFBZ0M7QUFDOUIsZUFDRSxNQUFDLHlDQUFELENBQU0sSUFBTjtBQUNFLGFBQUcsRUFBRUQsTUFBTSxDQUFDQyxLQURkO0FBRUUsaUJBQU8sRUFBRSxtQkFBTTtBQUNiLGdCQUFNQyxLQUFLLEdBQUdGLE1BQU0sQ0FBQ0csTUFBUCxHQUFnQkgsTUFBTSxDQUFDRyxNQUFQLEVBQWhCLEdBQWtDLEVBQWhEO0FBQ0FDLDBCQUFjLENBQUNGLEtBQUQsQ0FBZDtBQUNELFdBTEg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQU1FLE1BQUMsMkRBQUQ7QUFBVyxpQkFBTyxFQUFDLE1BQW5CO0FBQTBCLHdCQUFjLEVBQUMsY0FBekM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQUNFLE1BQUMsMkRBQUQ7QUFBVyxlQUFLLEVBQUMsR0FBakI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQUFzQkYsTUFBTSxDQUFDSyxJQUE3QixDQURGLEVBRUUsTUFBQywyREFBRDtBQUFXLGVBQUssRUFBQyxHQUFqQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQXNCTCxNQUFNLENBQUNNLEtBQTdCLENBRkYsQ0FORixDQURGO0FBYUQsT0FkRCxNQWNPO0FBQ0wsZUFDRSxNQUFDLHlDQUFELENBQU0sSUFBTjtBQUNFLGFBQUcsRUFBRU4sTUFBTSxDQUFDQyxLQURkO0FBRUUsaUJBQU8sRUFBRSxtQkFBTTtBQUNiRCxrQkFBTSxDQUFDRyxNQUFQLElBQWlCSCxNQUFNLENBQUNHLE1BQVAsQ0FBY0gsTUFBTSxDQUFDQyxLQUFyQixDQUFqQjtBQUNELFdBSkg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQU1FLE1BQUMsMkRBQUQ7QUFBVyxpQkFBTyxFQUFDLE1BQW5CO0FBQTBCLHdCQUFjLEVBQUMsY0FBekM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQUNFLE1BQUMsMkRBQUQ7QUFBVyxlQUFLLEVBQUMsR0FBakI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQUFzQkQsTUFBTSxDQUFDSyxJQUE3QixDQURGLEVBRUUsTUFBQywyREFBRDtBQUFXLGVBQUssRUFBQyxHQUFqQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQXNCTCxNQUFNLENBQUNNLEtBQTdCLENBRkYsQ0FORixDQURGO0FBYUQ7QUFDRixLQTlCQSxDQURILENBRGU7QUFBQSxHQUFqQjs7QUFvQ0EsU0FDRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDZDQUFEO0FBQVUsV0FBTyxFQUFFUixRQUFRLENBQUNELE9BQUQsQ0FBM0I7QUFBc0MsV0FBTyxFQUFFLENBQUMsT0FBRCxDQUEvQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywyQ0FBRDtBQUFRLFFBQUksRUFBQyxNQUFiO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsY0FDTyxNQUFDLDhEQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFEUCxDQURGLENBREYsQ0FERixDQURGO0FBV0QsQ0FoRE07S0FBTUQsYyIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC43MTVlZGY1MTQyYjAyNDI5MDZmYi5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xyXG5pbXBvcnQgeyBNZW51LCBEcm9wZG93biwgUm93LCBDb2wsIEJ1dHRvbiB9IGZyb20gJ2FudGQnO1xyXG5pbXBvcnQgeyBEb3duT3V0bGluZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XHJcblxyXG5pbXBvcnQgeyBPcHRpb25Qcm9wcyB9IGZyb20gJy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHsgQ3VzdG9tRGl2IH0gZnJvbSAnLi4vLi4vc3R5bGVkQ29tcG9uZW50cyc7XHJcblxyXG5leHBvcnQgaW50ZXJmYWNlIE1lbnVQcm9wcyB7XHJcbiAgb3B0aW9uczogT3B0aW9uUHJvcHNbXTtcclxufVxyXG5cclxuLy8gL1xyXG5jb25zb2xlLmxvZygnc3MnKVxyXG5leHBvcnQgY29uc3QgWm9vbWVkUGxvdE1lbnUgPSAoeyBvcHRpb25zIH06IE1lbnVQcm9wcykgPT4ge1xyXG4gIGNvbnN0IHBsb3RNZW51ID0gKG9wdGlvbnM6IE9wdGlvblByb3BzW10pID0+IChcclxuICAgIDxNZW51PlxyXG4gICAgICB7b3B0aW9ucy5tYXAoKG9wdGlvbjogT3B0aW9uUHJvcHMpID0+IHtcclxuICAgICAgICBpZiAob3B0aW9uLnZhbHVlID09PSAnb3ZlcmxheScpIHtcclxuICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgIDxNZW51Lkl0ZW1cclxuICAgICAgICAgICAgICBrZXk9e29wdGlvbi52YWx1ZX1cclxuICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICBjb25zdCBxdWVyeSA9IG9wdGlvbi5hY3Rpb24gPyBvcHRpb24uYWN0aW9uKCkgOiAnJztcclxuICAgICAgICAgICAgICAgIG9wZW5fYV9uZXdfdGFiKHF1ZXJ5IGFzIHN0cmluZylcclxuICAgICAgICAgICAgICB9fT5cclxuICAgICAgICAgICAgICA8Q3VzdG9tRGl2IGRpc3BsYXk9XCJmbGV4XCIganVzdGlmeWNvbnRlbnQ9XCJzcGFjZS1hcm91bmRcIj5cclxuICAgICAgICAgICAgICAgIDxDdXN0b21EaXYgc3BhY2U9XCIyXCI+e29wdGlvbi5pY29ufTwvQ3VzdG9tRGl2PlxyXG4gICAgICAgICAgICAgICAgPEN1c3RvbURpdiBzcGFjZT1cIjJcIj57b3B0aW9uLmxhYmVsfTwvQ3VzdG9tRGl2PlxyXG4gICAgICAgICAgICAgIDwvQ3VzdG9tRGl2PlxyXG4gICAgICAgICAgICA8L01lbnUuSXRlbT5cclxuICAgICAgICAgIClcclxuICAgICAgICB9IGVsc2Uge1xyXG4gICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgPE1lbnUuSXRlbVxyXG4gICAgICAgICAgICAgIGtleT17b3B0aW9uLnZhbHVlfVxyXG4gICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICAgIG9wdGlvbi5hY3Rpb24gJiYgb3B0aW9uLmFjdGlvbihvcHRpb24udmFsdWUpO1xyXG4gICAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAgID5cclxuICAgICAgICAgICAgICA8Q3VzdG9tRGl2IGRpc3BsYXk9XCJmbGV4XCIganVzdGlmeWNvbnRlbnQ9XCJzcGFjZS1hcm91bmRcIj5cclxuICAgICAgICAgICAgICAgIDxDdXN0b21EaXYgc3BhY2U9XCIyXCI+e29wdGlvbi5pY29ufTwvQ3VzdG9tRGl2PlxyXG4gICAgICAgICAgICAgICAgPEN1c3RvbURpdiBzcGFjZT1cIjJcIj57b3B0aW9uLmxhYmVsfTwvQ3VzdG9tRGl2PlxyXG4gICAgICAgICAgICAgIDwvQ3VzdG9tRGl2PlxyXG4gICAgICAgICAgICA8L01lbnUuSXRlbT5cclxuICAgICAgICAgIClcclxuICAgICAgICB9XHJcbiAgICAgIH0pfVxyXG4gICAgPC9NZW51PlxyXG4gICk7XHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8Um93PlxyXG4gICAgICA8Q29sPlxyXG4gICAgICAgIDxEcm9wZG93biBvdmVybGF5PXtwbG90TWVudShvcHRpb25zKX0gdHJpZ2dlcj17Wydob3ZlciddfT5cclxuICAgICAgICAgIDxCdXR0b24gdHlwZT1cImxpbmtcIj5cclxuICAgICAgICAgICAgTW9yZSA8RG93bk91dGxpbmVkIC8+XHJcbiAgICAgICAgICA8L0J1dHRvbj5cclxuICAgICAgICA8L0Ryb3Bkb3duPlxyXG4gICAgICA8L0NvbD5cclxuICAgIDwvUm93PlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=