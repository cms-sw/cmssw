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
/* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! next/link */ "./node_modules/next/link.js");
/* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(next_link__WEBPACK_IMPORTED_MODULE_4__);
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/plots/zoomedPlots/menu.tsx",
    _this = undefined;

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];





var ZoomedPlotMenu = function ZoomedPlotMenu(_ref) {
  var options = _ref.options;

  var plotMenu = function plotMenu(options) {
    return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 15,
        columnNumber: 5
      }
    }, options.map(function (option) {
      if (option.value === 'overlay') {
        return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"].Item, {
          key: option.value,
          onClick: function onClick() {
            option.action && option.action(option.value);
          },
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 19,
            columnNumber: 13
          }
        }, __jsx(next_link__WEBPACK_IMPORTED_MODULE_4___default.a, {
          href: "/index",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 25,
            columnNumber: 15
          }
        }, __jsx("a", {
          href: //@ts-ignore
          option.action && option.action(),
          target: "_blank",
          rel: "noreferrer",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 26,
            columnNumber: 15
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          display: "flex",
          justifycontent: "space-around",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 28,
            columnNumber: 17
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 29,
            columnNumber: 19
          }
        }, option.icon), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 30,
            columnNumber: 19
          }
        }, option.label)))));
      } else {
        return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"].Item, {
          key: option.value,
          onClick: function onClick() {
            option.action && option.action(option.value);
          },
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 39,
            columnNumber: 13
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          display: "flex",
          justifycontent: "space-around",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 45,
            columnNumber: 15
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 46,
            columnNumber: 17
          }
        }, option.icon), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 47,
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
      lineNumber: 57,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 58,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Dropdown"], {
    overlay: plotMenu(options),
    trigger: ['hover'],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 59,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 60,
      columnNumber: 11
    }
  }, "More ", __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["DownOutlined"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 61,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy9tZW51LnRzeCJdLCJuYW1lcyI6WyJab29tZWRQbG90TWVudSIsIm9wdGlvbnMiLCJwbG90TWVudSIsIm1hcCIsIm9wdGlvbiIsInZhbHVlIiwiYWN0aW9uIiwiaWNvbiIsImxhYmVsIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUdBO0FBQ0E7QUFNTyxJQUFNQSxjQUFjLEdBQUcsU0FBakJBLGNBQWlCLE9BQTRCO0FBQUEsTUFBekJDLE9BQXlCLFFBQXpCQSxPQUF5Qjs7QUFDeEQsTUFBTUMsUUFBUSxHQUFHLFNBQVhBLFFBQVcsQ0FBQ0QsT0FBRDtBQUFBLFdBQ2YsTUFBQyx5Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQ0dBLE9BQU8sQ0FBQ0UsR0FBUixDQUFZLFVBQUNDLE1BQUQsRUFBeUI7QUFDcEMsVUFBSUEsTUFBTSxDQUFDQyxLQUFQLEtBQWlCLFNBQXJCLEVBQWdDO0FBQzlCLGVBQ0UsTUFBQyx5Q0FBRCxDQUFNLElBQU47QUFDRSxhQUFHLEVBQUVELE1BQU0sQ0FBQ0MsS0FEZDtBQUVFLGlCQUFPLEVBQUUsbUJBQU07QUFDYkQsa0JBQU0sQ0FBQ0UsTUFBUCxJQUFpQkYsTUFBTSxDQUFDRSxNQUFQLENBQWNGLE1BQU0sQ0FBQ0MsS0FBckIsQ0FBakI7QUFDRCxXQUpIO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FNRSxNQUFDLGdEQUFEO0FBQU0sY0FBSSxFQUFDLFFBQVg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQUNBO0FBQUcsY0FBSSxFQUFFO0FBQ1BELGdCQUFNLENBQUNFLE1BQVAsSUFBaUNGLE1BQU0sQ0FBQ0UsTUFBUCxFQURuQztBQUM4RCxnQkFBTSxFQUFDLFFBRHJFO0FBQzhFLGFBQUcsRUFBQyxZQURsRjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBRUUsTUFBQywyREFBRDtBQUFXLGlCQUFPLEVBQUMsTUFBbkI7QUFBMEIsd0JBQWMsRUFBQyxjQUF6QztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQ0UsTUFBQywyREFBRDtBQUFXLGVBQUssRUFBQyxHQUFqQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQXNCRixNQUFNLENBQUNHLElBQTdCLENBREYsRUFFRSxNQUFDLDJEQUFEO0FBQVcsZUFBSyxFQUFDLEdBQWpCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FBc0JILE1BQU0sQ0FBQ0ksS0FBN0IsQ0FGRixDQUZGLENBREEsQ0FORixDQURGO0FBa0JELE9BbkJELE1Bb0JLO0FBQ0gsZUFDRSxNQUFDLHlDQUFELENBQU0sSUFBTjtBQUNFLGFBQUcsRUFBRUosTUFBTSxDQUFDQyxLQURkO0FBRUUsaUJBQU8sRUFBRSxtQkFBTTtBQUNiRCxrQkFBTSxDQUFDRSxNQUFQLElBQWlCRixNQUFNLENBQUNFLE1BQVAsQ0FBY0YsTUFBTSxDQUFDQyxLQUFyQixDQUFqQjtBQUNELFdBSkg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQU1FLE1BQUMsMkRBQUQ7QUFBVyxpQkFBTyxFQUFDLE1BQW5CO0FBQTBCLHdCQUFjLEVBQUMsY0FBekM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQUNFLE1BQUMsMkRBQUQ7QUFBVyxlQUFLLEVBQUMsR0FBakI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQUFzQkQsTUFBTSxDQUFDRyxJQUE3QixDQURGLEVBRUUsTUFBQywyREFBRDtBQUFXLGVBQUssRUFBQyxHQUFqQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQXNCSCxNQUFNLENBQUNJLEtBQTdCLENBRkYsQ0FORixDQURGO0FBYUQ7QUFDRixLQXBDQSxDQURILENBRGU7QUFBQSxHQUFqQjs7QUEwQ0EsU0FDRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLHdDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsS0FDRSxNQUFDLDZDQUFEO0FBQVUsV0FBTyxFQUFFTixRQUFRLENBQUNELE9BQUQsQ0FBM0I7QUFBc0MsV0FBTyxFQUFFLENBQUMsT0FBRCxDQUEvQztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywyQ0FBRDtBQUFRLFFBQUksRUFBQyxNQUFiO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsY0FDTyxNQUFDLDhEQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFEUCxDQURGLENBREYsQ0FERixDQURGO0FBV0QsQ0F0RE07S0FBTUQsYyIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC42YWU2MzAyZDFiNmQxY2E3NGNlZS5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0ICogYXMgUmVhY3QgZnJvbSAncmVhY3QnO1xyXG5pbXBvcnQgeyBNZW51LCBEcm9wZG93biwgUm93LCBDb2wsIEJ1dHRvbiB9IGZyb20gJ2FudGQnO1xyXG5pbXBvcnQgeyBEb3duT3V0bGluZWQgfSBmcm9tICdAYW50LWRlc2lnbi9pY29ucyc7XHJcblxyXG5pbXBvcnQgeyBPcHRpb25Qcm9wcyB9IGZyb20gJy4uLy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9pbnRlcmZhY2VzJztcclxuaW1wb3J0IHsgQ3VzdG9tRGl2IH0gZnJvbSAnLi4vLi4vc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCBMaW5rIGZyb20gJ25leHQvbGluayc7XHJcblxyXG5leHBvcnQgaW50ZXJmYWNlIE1lbnVQcm9wcyB7XHJcbiAgb3B0aW9uczogT3B0aW9uUHJvcHNbXTtcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IFpvb21lZFBsb3RNZW51ID0gKHsgb3B0aW9ucyB9OiBNZW51UHJvcHMpID0+IHtcclxuICBjb25zdCBwbG90TWVudSA9IChvcHRpb25zOiBPcHRpb25Qcm9wc1tdKSA9PiAoXHJcbiAgICA8TWVudT5cclxuICAgICAge29wdGlvbnMubWFwKChvcHRpb246IE9wdGlvblByb3BzKSA9PiB7XHJcbiAgICAgICAgaWYgKG9wdGlvbi52YWx1ZSA9PT0gJ292ZXJsYXknKSB7XHJcbiAgICAgICAgICByZXR1cm4gKFxyXG4gICAgICAgICAgICA8TWVudS5JdGVtXHJcbiAgICAgICAgICAgICAga2V5PXtvcHRpb24udmFsdWV9XHJcbiAgICAgICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICAgICAgb3B0aW9uLmFjdGlvbiAmJiBvcHRpb24uYWN0aW9uKG9wdGlvbi52YWx1ZSk7XHJcbiAgICAgICAgICAgICAgfX1cclxuICAgICAgICAgICAgPlxyXG4gICAgICAgICAgICAgIDxMaW5rIGhyZWY9XCIvaW5kZXhcIj5cclxuICAgICAgICAgICAgICA8YSBocmVmPXsvL0B0cy1pZ25vcmVcclxuICAgICAgICAgICAgICAgIG9wdGlvbi5hY3Rpb24gYXMgKCkgPT4gc3RyaW5nICYmIG9wdGlvbi5hY3Rpb24oKSBhcyBzdHJpbmd9IHRhcmdldD1cIl9ibGFua1wiIHJlbD1cIm5vcmVmZXJyZXJcIiA+XHJcbiAgICAgICAgICAgICAgICA8Q3VzdG9tRGl2IGRpc3BsYXk9XCJmbGV4XCIganVzdGlmeWNvbnRlbnQ9XCJzcGFjZS1hcm91bmRcIj5cclxuICAgICAgICAgICAgICAgICAgPEN1c3RvbURpdiBzcGFjZT1cIjJcIj57b3B0aW9uLmljb259PC9DdXN0b21EaXY+XHJcbiAgICAgICAgICAgICAgICAgIDxDdXN0b21EaXYgc3BhY2U9XCIyXCI+e29wdGlvbi5sYWJlbH08L0N1c3RvbURpdj5cclxuICAgICAgICAgICAgICAgIDwvQ3VzdG9tRGl2PlxyXG4gICAgICAgICAgICAgIDwvYT5cclxuICAgICAgICAgICAgICA8L0xpbms+XHJcbiAgICAgICAgICAgIDwvTWVudS5JdGVtPlxyXG4gICAgICAgICAgKVxyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgIDxNZW51Lkl0ZW1cclxuICAgICAgICAgICAgICBrZXk9e29wdGlvbi52YWx1ZX1cclxuICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICBvcHRpb24uYWN0aW9uICYmIG9wdGlvbi5hY3Rpb24ob3B0aW9uLnZhbHVlKTtcclxuICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICA+XHJcbiAgICAgICAgICAgICAgPEN1c3RvbURpdiBkaXNwbGF5PVwiZmxleFwiIGp1c3RpZnljb250ZW50PVwic3BhY2UtYXJvdW5kXCI+XHJcbiAgICAgICAgICAgICAgICA8Q3VzdG9tRGl2IHNwYWNlPVwiMlwiPntvcHRpb24uaWNvbn08L0N1c3RvbURpdj5cclxuICAgICAgICAgICAgICAgIDxDdXN0b21EaXYgc3BhY2U9XCIyXCI+e29wdGlvbi5sYWJlbH08L0N1c3RvbURpdj5cclxuICAgICAgICAgICAgICA8L0N1c3RvbURpdj5cclxuICAgICAgICAgICAgPC9NZW51Lkl0ZW0+XHJcbiAgICAgICAgICApXHJcbiAgICAgICAgfVxyXG4gICAgICB9KX1cclxuICAgIDwvTWVudT5cclxuICApO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPFJvdz5cclxuICAgICAgPENvbD5cclxuICAgICAgICA8RHJvcGRvd24gb3ZlcmxheT17cGxvdE1lbnUob3B0aW9ucyl9IHRyaWdnZXI9e1snaG92ZXInXX0+XHJcbiAgICAgICAgICA8QnV0dG9uIHR5cGU9XCJsaW5rXCI+XHJcbiAgICAgICAgICAgIE1vcmUgPERvd25PdXRsaW5lZCAvPlxyXG4gICAgICAgICAgPC9CdXR0b24+XHJcbiAgICAgICAgPC9Ecm9wZG93bj5cclxuICAgICAgPC9Db2w+XHJcbiAgICA8L1Jvdz5cclxuICApO1xyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9