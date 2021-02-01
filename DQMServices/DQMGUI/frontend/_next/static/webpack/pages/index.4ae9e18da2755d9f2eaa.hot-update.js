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




var ZoomedPlotMenu = function ZoomedPlotMenu(_ref) {
  var options = _ref.options;

  var plotMenu = function plotMenu(options) {
    return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 14,
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
            lineNumber: 18,
            columnNumber: 13
          }
        }, __jsx("a", {
          href: //@ts-ignore
          option.action && option.action(),
          target: "_blank",
          rel: "noreferrer",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 25,
            columnNumber: 15
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          display: "flex",
          justifycontent: "space-around",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 27,
            columnNumber: 17
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 28,
            columnNumber: 19
          }
        }, option.icon), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 29,
            columnNumber: 19
          }
        }, option.label))));
      } else {
        return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"].Item, {
          key: option.value,
          onClick: function onClick() {
            option.action && option.action(option.value);
          },
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 37,
            columnNumber: 13
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          display: "flex",
          justifycontent: "space-around",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 43,
            columnNumber: 15
          }
        }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 44,
            columnNumber: 17
          }
        }, option.icon), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
          space: "2",
          __self: _this,
          __source: {
            fileName: _jsxFileName,
            lineNumber: 45,
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
      lineNumber: 55,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 56,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Dropdown"], {
    overlay: plotMenu(options),
    trigger: ['hover'],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 57,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 58,
      columnNumber: 11
    }
  }, "More ", __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["DownOutlined"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 59,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy9tZW51LnRzeCJdLCJuYW1lcyI6WyJab29tZWRQbG90TWVudSIsIm9wdGlvbnMiLCJwbG90TWVudSIsIm1hcCIsIm9wdGlvbiIsInZhbHVlIiwiYWN0aW9uIiwiaWNvbiIsImxhYmVsIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFHQTtBQU1PLElBQU1BLGNBQWMsR0FBRyxTQUFqQkEsY0FBaUIsT0FBNEI7QUFBQSxNQUF6QkMsT0FBeUIsUUFBekJBLE9BQXlCOztBQUN4RCxNQUFNQyxRQUFRLEdBQUcsU0FBWEEsUUFBVyxDQUFDRCxPQUFEO0FBQUEsV0FDZixNQUFDLHlDQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FDR0EsT0FBTyxDQUFDRSxHQUFSLENBQVksVUFBQ0MsTUFBRCxFQUF5QjtBQUNwQyxVQUFJQSxNQUFNLENBQUNDLEtBQVAsS0FBaUIsU0FBckIsRUFBZ0M7QUFDOUIsZUFDRSxNQUFDLHlDQUFELENBQU0sSUFBTjtBQUNFLGFBQUcsRUFBRUQsTUFBTSxDQUFDQyxLQURkO0FBRUUsaUJBQU8sRUFBRSxtQkFBTTtBQUNiRCxrQkFBTSxDQUFDRSxNQUFQLElBQWlCRixNQUFNLENBQUNFLE1BQVAsQ0FBY0YsTUFBTSxDQUFDQyxLQUFyQixDQUFqQjtBQUNELFdBSkg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQU9FO0FBQUcsY0FBSSxFQUFFO0FBQ1BELGdCQUFNLENBQUNFLE1BQVAsSUFBaUNGLE1BQU0sQ0FBQ0UsTUFBUCxFQURuQztBQUM4RCxnQkFBTSxFQUFDLFFBRHJFO0FBQzhFLGFBQUcsRUFBQyxZQURsRjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBRUUsTUFBQywyREFBRDtBQUFXLGlCQUFPLEVBQUMsTUFBbkI7QUFBMEIsd0JBQWMsRUFBQyxjQUF6QztBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQ0UsTUFBQywyREFBRDtBQUFXLGVBQUssRUFBQyxHQUFqQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFdBQXNCRixNQUFNLENBQUNHLElBQTdCLENBREYsRUFFRSxNQUFDLDJEQUFEO0FBQVcsZUFBSyxFQUFDLEdBQWpCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FBc0JILE1BQU0sQ0FBQ0ksS0FBN0IsQ0FGRixDQUZGLENBUEYsQ0FERjtBQWlCRCxPQWxCRCxNQW1CSztBQUNILGVBQ0UsTUFBQyx5Q0FBRCxDQUFNLElBQU47QUFDRSxhQUFHLEVBQUVKLE1BQU0sQ0FBQ0MsS0FEZDtBQUVFLGlCQUFPLEVBQUUsbUJBQU07QUFDYkQsa0JBQU0sQ0FBQ0UsTUFBUCxJQUFpQkYsTUFBTSxDQUFDRSxNQUFQLENBQWNGLE1BQU0sQ0FBQ0MsS0FBckIsQ0FBakI7QUFDRCxXQUpIO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FNRSxNQUFDLDJEQUFEO0FBQVcsaUJBQU8sRUFBQyxNQUFuQjtBQUEwQix3QkFBYyxFQUFDLGNBQXpDO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FDRSxNQUFDLDJEQUFEO0FBQVcsZUFBSyxFQUFDLEdBQWpCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsV0FBc0JELE1BQU0sQ0FBQ0csSUFBN0IsQ0FERixFQUVFLE1BQUMsMkRBQUQ7QUFBVyxlQUFLLEVBQUMsR0FBakI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxXQUFzQkgsTUFBTSxDQUFDSSxLQUE3QixDQUZGLENBTkYsQ0FERjtBQWFEO0FBQ0YsS0FuQ0EsQ0FESCxDQURlO0FBQUEsR0FBakI7O0FBeUNBLFNBQ0UsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyw2Q0FBRDtBQUFVLFdBQU8sRUFBRU4sUUFBUSxDQUFDRCxPQUFELENBQTNCO0FBQXNDLFdBQU8sRUFBRSxDQUFDLE9BQUQsQ0FBL0M7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsMkNBQUQ7QUFBUSxRQUFJLEVBQUMsTUFBYjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLGNBQ08sTUFBQyw4REFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBRFAsQ0FERixDQURGLENBREYsQ0FERjtBQVdELENBckRNO0tBQU1ELGMiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguNGFlOWUxOGRhMjc1NWQ5ZjJlYWEuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgTWVudSwgRHJvcGRvd24sIFJvdywgQ29sLCBCdXR0b24gfSBmcm9tICdhbnRkJztcclxuaW1wb3J0IHsgRG93bk91dGxpbmVkIH0gZnJvbSAnQGFudC1kZXNpZ24vaWNvbnMnO1xyXG5cclxuaW1wb3J0IHsgT3B0aW9uUHJvcHMgfSBmcm9tICcuLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IEN1c3RvbURpdiB9IGZyb20gJy4uLy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5cclxuZXhwb3J0IGludGVyZmFjZSBNZW51UHJvcHMge1xyXG4gIG9wdGlvbnM6IE9wdGlvblByb3BzW107XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBab29tZWRQbG90TWVudSA9ICh7IG9wdGlvbnMgfTogTWVudVByb3BzKSA9PiB7XHJcbiAgY29uc3QgcGxvdE1lbnUgPSAob3B0aW9uczogT3B0aW9uUHJvcHNbXSkgPT4gKFxyXG4gICAgPE1lbnU+XHJcbiAgICAgIHtvcHRpb25zLm1hcCgob3B0aW9uOiBPcHRpb25Qcm9wcykgPT4ge1xyXG4gICAgICAgIGlmIChvcHRpb24udmFsdWUgPT09ICdvdmVybGF5Jykge1xyXG4gICAgICAgICAgcmV0dXJuIChcclxuICAgICAgICAgICAgPE1lbnUuSXRlbVxyXG4gICAgICAgICAgICAgIGtleT17b3B0aW9uLnZhbHVlfVxyXG4gICAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+IHtcclxuICAgICAgICAgICAgICAgIG9wdGlvbi5hY3Rpb24gJiYgb3B0aW9uLmFjdGlvbihvcHRpb24udmFsdWUpO1xyXG4gICAgICAgICAgICAgIH19XHJcbiAgICAgICAgICAgID5cclxuXHJcbiAgICAgICAgICAgICAgPGEgaHJlZj17Ly9AdHMtaWdub3JlXHJcbiAgICAgICAgICAgICAgICBvcHRpb24uYWN0aW9uIGFzICgpID0+IHN0cmluZyAmJiBvcHRpb24uYWN0aW9uKCkgYXMgc3RyaW5nfSB0YXJnZXQ9XCJfYmxhbmtcIiByZWw9XCJub3JlZmVycmVyXCIgPlxyXG4gICAgICAgICAgICAgICAgPEN1c3RvbURpdiBkaXNwbGF5PVwiZmxleFwiIGp1c3RpZnljb250ZW50PVwic3BhY2UtYXJvdW5kXCI+XHJcbiAgICAgICAgICAgICAgICAgIDxDdXN0b21EaXYgc3BhY2U9XCIyXCI+e29wdGlvbi5pY29ufTwvQ3VzdG9tRGl2PlxyXG4gICAgICAgICAgICAgICAgICA8Q3VzdG9tRGl2IHNwYWNlPVwiMlwiPntvcHRpb24ubGFiZWx9PC9DdXN0b21EaXY+XHJcbiAgICAgICAgICAgICAgICA8L0N1c3RvbURpdj5cclxuICAgICAgICAgICAgICA8L2E+XHJcbiAgICAgICAgICAgIDwvTWVudS5JdGVtPlxyXG4gICAgICAgICAgKVxyXG4gICAgICAgIH1cclxuICAgICAgICBlbHNlIHtcclxuICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgIDxNZW51Lkl0ZW1cclxuICAgICAgICAgICAgICBrZXk9e29wdGlvbi52YWx1ZX1cclxuICAgICAgICAgICAgICBvbkNsaWNrPXsoKSA9PiB7XHJcbiAgICAgICAgICAgICAgICBvcHRpb24uYWN0aW9uICYmIG9wdGlvbi5hY3Rpb24ob3B0aW9uLnZhbHVlKTtcclxuICAgICAgICAgICAgICB9fVxyXG4gICAgICAgICAgICA+XHJcbiAgICAgICAgICAgICAgPEN1c3RvbURpdiBkaXNwbGF5PVwiZmxleFwiIGp1c3RpZnljb250ZW50PVwic3BhY2UtYXJvdW5kXCI+XHJcbiAgICAgICAgICAgICAgICA8Q3VzdG9tRGl2IHNwYWNlPVwiMlwiPntvcHRpb24uaWNvbn08L0N1c3RvbURpdj5cclxuICAgICAgICAgICAgICAgIDxDdXN0b21EaXYgc3BhY2U9XCIyXCI+e29wdGlvbi5sYWJlbH08L0N1c3RvbURpdj5cclxuICAgICAgICAgICAgICA8L0N1c3RvbURpdj5cclxuICAgICAgICAgICAgPC9NZW51Lkl0ZW0+XHJcbiAgICAgICAgICApXHJcbiAgICAgICAgfVxyXG4gICAgICB9KX1cclxuICAgIDwvTWVudT5cclxuICApO1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPFJvdz5cclxuICAgICAgPENvbD5cclxuICAgICAgICA8RHJvcGRvd24gb3ZlcmxheT17cGxvdE1lbnUob3B0aW9ucyl9IHRyaWdnZXI9e1snaG92ZXInXX0+XHJcbiAgICAgICAgICA8QnV0dG9uIHR5cGU9XCJsaW5rXCI+XHJcbiAgICAgICAgICAgIE1vcmUgPERvd25PdXRsaW5lZCAvPlxyXG4gICAgICAgICAgPC9CdXR0b24+XHJcbiAgICAgICAgPC9Ecm9wZG93bj5cclxuICAgICAgPC9Db2w+XHJcbiAgICA8L1Jvdz5cclxuICApO1xyXG59O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9