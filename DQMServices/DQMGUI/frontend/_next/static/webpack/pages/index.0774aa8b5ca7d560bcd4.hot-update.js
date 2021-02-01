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
      return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Menu"].Item, {
        key: option.value,
        onClick: function onClick() {
          option.action && option.action(option.value);
        },
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 17,
          columnNumber: 9
        }
      }, __jsx(next_link__WEBPACK_IMPORTED_MODULE_4___default.a, {
        href: "/index",
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 23,
          columnNumber: 11
        }
      }, __jsx("a", {
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
          lineNumber: 25,
          columnNumber: 15
        }
      }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
        space: "2",
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 26,
          columnNumber: 17
        }
      }, option.icon), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_3__["CustomDiv"], {
        space: "2",
        __self: _this,
        __source: {
          fileName: _jsxFileName,
          lineNumber: 27,
          columnNumber: 17
        }
      }, option.label)))));
    }));
  };

  return __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Row"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 37,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Col"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 38,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Dropdown"], {
    overlay: plotMenu(options),
    trigger: ['hover'],
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 39,
      columnNumber: 9
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_1__["Button"], {
    type: "link",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 40,
      columnNumber: 11
    }
  }, "More ", __jsx(_ant_design_icons__WEBPACK_IMPORTED_MODULE_2__["DownOutlined"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 41,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9wbG90cy96b29tZWRQbG90cy9tZW51LnRzeCJdLCJuYW1lcyI6WyJab29tZWRQbG90TWVudSIsIm9wdGlvbnMiLCJwbG90TWVudSIsIm1hcCIsIm9wdGlvbiIsInZhbHVlIiwiYWN0aW9uIiwiaWNvbiIsImxhYmVsIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFDQTtBQUdBO0FBQ0E7QUFNTyxJQUFNQSxjQUFjLEdBQUcsU0FBakJBLGNBQWlCLE9BQTRCO0FBQUEsTUFBekJDLE9BQXlCLFFBQXpCQSxPQUF5Qjs7QUFDeEQsTUFBTUMsUUFBUSxHQUFHLFNBQVhBLFFBQVcsQ0FBQ0QsT0FBRDtBQUFBLFdBQ2YsTUFBQyx5Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQ0dBLE9BQU8sQ0FBQ0UsR0FBUixDQUFZLFVBQUNDLE1BQUQ7QUFBQSxhQUNYLE1BQUMseUNBQUQsQ0FBTSxJQUFOO0FBQ0UsV0FBRyxFQUFFQSxNQUFNLENBQUNDLEtBRGQ7QUFFRSxlQUFPLEVBQUUsbUJBQU07QUFDYkQsZ0JBQU0sQ0FBQ0UsTUFBUCxJQUFpQkYsTUFBTSxDQUFDRSxNQUFQLENBQWNGLE1BQU0sQ0FBQ0MsS0FBckIsQ0FBakI7QUFDRCxTQUpIO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsU0FNRSxNQUFDLGdEQUFEO0FBQU0sWUFBSSxFQUFDLFFBQVg7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxTQUNFO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsU0FDRSxNQUFDLDJEQUFEO0FBQVcsZUFBTyxFQUFDLE1BQW5CO0FBQTBCLHNCQUFjLEVBQUMsY0FBekM7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxTQUNFLE1BQUMsMkRBQUQ7QUFBVyxhQUFLLEVBQUMsR0FBakI7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxTQUFzQkQsTUFBTSxDQUFDRyxJQUE3QixDQURGLEVBRUUsTUFBQywyREFBRDtBQUFXLGFBQUssRUFBQyxHQUFqQjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLFNBQXNCSCxNQUFNLENBQUNJLEtBQTdCLENBRkYsQ0FERixDQURGLENBTkYsQ0FEVztBQUFBLEtBQVosQ0FESCxDQURlO0FBQUEsR0FBakI7O0FBc0JBLFNBQ0UsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyx3Q0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQyw2Q0FBRDtBQUFVLFdBQU8sRUFBRU4sUUFBUSxDQUFDRCxPQUFELENBQTNCO0FBQXNDLFdBQU8sRUFBRSxDQUFDLE9BQUQsQ0FBL0M7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsMkNBQUQ7QUFBUSxRQUFJLEVBQUMsTUFBYjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLGNBQ08sTUFBQyw4REFBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBRFAsQ0FERixDQURGLENBREYsQ0FERjtBQVdELENBbENNO0tBQU1ELGMiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguMDc3NGFhOGI1Y2E3ZDU2MGJjZDQuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCAqIGFzIFJlYWN0IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgTWVudSwgRHJvcGRvd24sIFJvdywgQ29sLCBCdXR0b24gfSBmcm9tICdhbnRkJztcclxuaW1wb3J0IHsgRG93bk91dGxpbmVkIH0gZnJvbSAnQGFudC1kZXNpZ24vaWNvbnMnO1xyXG5cclxuaW1wb3J0IHsgT3B0aW9uUHJvcHMgfSBmcm9tICcuLi8uLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcyc7XHJcbmltcG9ydCB7IEN1c3RvbURpdiB9IGZyb20gJy4uLy4uL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgTGluayBmcm9tICduZXh0L2xpbmsnO1xyXG5cclxuZXhwb3J0IGludGVyZmFjZSBNZW51UHJvcHMge1xyXG4gIG9wdGlvbnM6IE9wdGlvblByb3BzW107XHJcbn1cclxuXHJcbmV4cG9ydCBjb25zdCBab29tZWRQbG90TWVudSA9ICh7IG9wdGlvbnMgfTogTWVudVByb3BzKSA9PiB7XHJcbiAgY29uc3QgcGxvdE1lbnUgPSAob3B0aW9uczogT3B0aW9uUHJvcHNbXSkgPT4gKFxyXG4gICAgPE1lbnU+XHJcbiAgICAgIHtvcHRpb25zLm1hcCgob3B0aW9uOiBPcHRpb25Qcm9wcykgPT4gKFxyXG4gICAgICAgIDxNZW51Lkl0ZW1cclxuICAgICAgICAgIGtleT17b3B0aW9uLnZhbHVlfVxyXG4gICAgICAgICAgb25DbGljaz17KCkgPT4ge1xyXG4gICAgICAgICAgICBvcHRpb24uYWN0aW9uICYmIG9wdGlvbi5hY3Rpb24ob3B0aW9uLnZhbHVlKTtcclxuICAgICAgICAgIH19XHJcbiAgICAgICAgPlxyXG4gICAgICAgICAgPExpbmsgaHJlZj1cIi9pbmRleFwiPlxyXG4gICAgICAgICAgICA8YT5cclxuICAgICAgICAgICAgICA8Q3VzdG9tRGl2IGRpc3BsYXk9XCJmbGV4XCIganVzdGlmeWNvbnRlbnQ9XCJzcGFjZS1hcm91bmRcIj5cclxuICAgICAgICAgICAgICAgIDxDdXN0b21EaXYgc3BhY2U9XCIyXCI+e29wdGlvbi5pY29ufTwvQ3VzdG9tRGl2PlxyXG4gICAgICAgICAgICAgICAgPEN1c3RvbURpdiBzcGFjZT1cIjJcIj57b3B0aW9uLmxhYmVsfTwvQ3VzdG9tRGl2PlxyXG4gICAgICAgICAgICAgIDwvQ3VzdG9tRGl2PlxyXG4gICAgICAgICAgICA8L2E+XHJcbiAgICAgICAgICA8L0xpbms+XHJcbiAgICAgICAgPC9NZW51Lkl0ZW0+XHJcbiAgICAgICkpfVxyXG4gICAgPC9NZW51PlxyXG4gICk7XHJcblxyXG4gIHJldHVybiAoXHJcbiAgICA8Um93PlxyXG4gICAgICA8Q29sPlxyXG4gICAgICAgIDxEcm9wZG93biBvdmVybGF5PXtwbG90TWVudShvcHRpb25zKX0gdHJpZ2dlcj17Wydob3ZlciddfT5cclxuICAgICAgICAgIDxCdXR0b24gdHlwZT1cImxpbmtcIj5cclxuICAgICAgICAgICAgTW9yZSA8RG93bk91dGxpbmVkIC8+XHJcbiAgICAgICAgICA8L0J1dHRvbj5cclxuICAgICAgICA8L0Ryb3Bkb3duPlxyXG4gICAgICA8L0NvbD5cclxuICAgIDwvUm93PlxyXG4gICk7XHJcbn07XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=