webpackHotUpdate_N_E("pages/index",{

/***/ "./components/plots/plot/plotSearch/index.tsx":
false,

/***/ "./components/workspaces/index.tsx":
false,

/***/ "./containers/display/header.tsx":
/*!***************************************!*\
  !*** ./containers/display/header.tsx ***!
  \***************************************/
/*! exports provided: Header */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Header", function() { return Header; });
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _utils_pages__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../../utils/pages */ "./utils/pages/index.tsx");
/* harmony import */ var _components_runInfo__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../../components/runInfo */ "./components/runInfo/index.tsx");
/* harmony import */ var _components_navigation_composedSearch__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../components/navigation/composedSearch */ "./components/navigation/composedSearch.tsx");
/* harmony import */ var _components_Nav__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../components/Nav */ "./components/Nav.tsx");
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/containers/display/header.tsx",
    _this = undefined;

var __jsx = react__WEBPACK_IMPORTED_MODULE_0__["createElement"];






var Header = function Header(_ref) {
  var isDatasetAndRunNumberSelected = _ref.isDatasetAndRunNumberSelected,
      query = _ref.query;
  return __jsx(antd__WEBPACK_IMPORTED_MODULE_5__["Col"], {
    style: {
      width: '100%'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 23,
      columnNumber: 5
    }
  }, //if all full set is selected: dataset name and run number, then regular search field is not visible.
  //Instead, run and dataset browser is is displayed.
  //Regular search fields are displayed just in the main page.
  isDatasetAndRunNumberSelected ? __jsx(antd__WEBPACK_IMPORTED_MODULE_5__["Col"], {
    style: {
      display: 'flex',
      alignItems: 'center',
      width: '100%'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 29,
      columnNumber: 11
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_5__["Col"], {
    style: {
      display: 'flex',
      alignItems: 'center',
      "float": 'left',
      padding: 8
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 30,
      columnNumber: 11
    }
  }, __jsx(_components_runInfo__WEBPACK_IMPORTED_MODULE_2__["RunInfo"], {
    query: query,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 31,
      columnNumber: 13
    }
  }), __jsx(_components_navigation_composedSearch__WEBPACK_IMPORTED_MODULE_3__["ComposedSearch"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 32,
      columnNumber: 13
    }
  }))) : __jsx(react__WEBPACK_IMPORTED_MODULE_0__["Fragment"], null, __jsx(_components_Nav__WEBPACK_IMPORTED_MODULE_4__["default"], {
    initial_search_run_number: query.search_run_number,
    initial_search_dataset_name: query.search_dataset_name,
    handler: _utils_pages__WEBPACK_IMPORTED_MODULE_1__["navigationHandler"],
    type: "top",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 41,
      columnNumber: 15
    }
  })));
};
_c = Header;

var _c;

$RefreshReg$(_c, "Header");

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

/***/ "./workspaces/online.ts":
false

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29udGFpbmVycy9kaXNwbGF5L2hlYWRlci50c3giXSwibmFtZXMiOlsiSGVhZGVyIiwiaXNEYXRhc2V0QW5kUnVuTnVtYmVyU2VsZWN0ZWQiLCJxdWVyeSIsIndpZHRoIiwiZGlzcGxheSIsImFsaWduSXRlbXMiLCJwYWRkaW5nIiwic2VhcmNoX3J1bl9udW1iZXIiLCJzZWFyY2hfZGF0YXNldF9uYW1lIiwibmF2aWdhdGlvbkhhbmRsZXIiXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQUVBO0FBVU8sSUFBTUEsTUFBTSxHQUFHLFNBQVRBLE1BQVMsT0FHSDtBQUFBLE1BRmpCQyw2QkFFaUIsUUFGakJBLDZCQUVpQjtBQUFBLE1BRGpCQyxLQUNpQixRQURqQkEsS0FDaUI7QUFDakIsU0FDRSxNQUFDLHdDQUFEO0FBQUssU0FBSyxFQUFFO0FBQUNDLFdBQUssRUFBRTtBQUFSLEtBQVo7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUVJO0FBQ0E7QUFDQTtBQUNBRiwrQkFBNkIsR0FDM0IsTUFBQyx3Q0FBRDtBQUFLLFNBQUssRUFBRTtBQUFDRyxhQUFPLEVBQUUsTUFBVjtBQUFrQkMsZ0JBQVUsRUFBRSxRQUE5QjtBQUF3Q0YsV0FBSyxFQUFFO0FBQS9DLEtBQVo7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNBLE1BQUMsd0NBQUQ7QUFBSyxTQUFLLEVBQUU7QUFBRUMsYUFBTyxFQUFFLE1BQVg7QUFBbUJDLGdCQUFVLEVBQUUsUUFBL0I7QUFBd0MsZUFBTyxNQUEvQztBQUF1REMsYUFBTyxFQUFFO0FBQWhFLEtBQVo7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxLQUNFLE1BQUMsMkRBQUQ7QUFBUyxTQUFLLEVBQUVKLEtBQWhCO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixFQUVFLE1BQUMsb0ZBQUQ7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQSxJQUZGLENBREEsQ0FEMkIsR0FZekIsNERBQ0UsTUFBQyx1REFBRDtBQUNFLDZCQUF5QixFQUFFQSxLQUFLLENBQUNLLGlCQURuQztBQUVFLCtCQUEyQixFQUFFTCxLQUFLLENBQUNNLG1CQUZyQztBQUdFLFdBQU8sRUFBRUMsOERBSFg7QUFJRSxRQUFJLEVBQUMsS0FKUDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0FqQlIsQ0FERjtBQThCRCxDQWxDTTtLQUFNVCxNIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmFkYTUxMjk1ZDA0ZGMwOWIyYjkwLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCc7XHJcblxyXG5pbXBvcnQgeyBuYXZpZ2F0aW9uSGFuZGxlciB9IGZyb20gJy4uLy4uL3V0aWxzL3BhZ2VzJztcclxuaW1wb3J0IHsgUnVuSW5mbyB9IGZyb20gJy4uLy4uL2NvbXBvbmVudHMvcnVuSW5mbyc7XHJcbmltcG9ydCB7IENvbXBvc2VkU2VhcmNoIH0gZnJvbSAnLi4vLi4vY29tcG9uZW50cy9uYXZpZ2F0aW9uL2NvbXBvc2VkU2VhcmNoJztcclxuaW1wb3J0IE5hdiBmcm9tICcuLi8uLi9jb21wb25lbnRzL05hdic7XHJcbmltcG9ydCB7IFF1ZXJ5UHJvcHMgfSBmcm9tICcuL2ludGVyZmFjZXMnO1xyXG5pbXBvcnQgeyBDb2wgfSBmcm9tICdhbnRkJztcclxuaW1wb3J0IHsgV3JhcHBlckRpdiB9IGZyb20gJy4vc3R5bGVkQ29tcG9uZW50cyc7XHJcbmltcG9ydCBXb3Jrc3BhY2VzIGZyb20gJy4uLy4uL2NvbXBvbmVudHMvd29ya3NwYWNlcyc7XHJcbmltcG9ydCB7IFBsb3RTZWFyY2ggfSBmcm9tICcuLi8uLi9jb21wb25lbnRzL3Bsb3RzL3Bsb3QvcGxvdFNlYXJjaCc7XHJcblxyXG5pbnRlcmZhY2UgSGVhZGVyUHJvcHMge1xyXG4gIGlzRGF0YXNldEFuZFJ1bk51bWJlclNlbGVjdGVkOiBib29sZWFuO1xyXG4gIHF1ZXJ5OiBRdWVyeVByb3BzO1xyXG59XHJcblxyXG5leHBvcnQgY29uc3QgSGVhZGVyID0gKHtcclxuICBpc0RhdGFzZXRBbmRSdW5OdW1iZXJTZWxlY3RlZCxcclxuICBxdWVyeSxcclxufTogSGVhZGVyUHJvcHMpID0+IHtcclxuICByZXR1cm4gKFxyXG4gICAgPENvbCBzdHlsZT17e3dpZHRoOiAnMTAwJSd9fT5cclxuICAgICAge1xyXG4gICAgICAgIC8vaWYgYWxsIGZ1bGwgc2V0IGlzIHNlbGVjdGVkOiBkYXRhc2V0IG5hbWUgYW5kIHJ1biBudW1iZXIsIHRoZW4gcmVndWxhciBzZWFyY2ggZmllbGQgaXMgbm90IHZpc2libGUuXHJcbiAgICAgICAgLy9JbnN0ZWFkLCBydW4gYW5kIGRhdGFzZXQgYnJvd3NlciBpcyBpcyBkaXNwbGF5ZWQuXHJcbiAgICAgICAgLy9SZWd1bGFyIHNlYXJjaCBmaWVsZHMgYXJlIGRpc3BsYXllZCBqdXN0IGluIHRoZSBtYWluIHBhZ2UuXHJcbiAgICAgICAgaXNEYXRhc2V0QW5kUnVuTnVtYmVyU2VsZWN0ZWQgPyAoXHJcbiAgICAgICAgICA8Q29sIHN0eWxlPXt7ZGlzcGxheTogJ2ZsZXgnLCBhbGlnbkl0ZW1zOiAnY2VudGVyJywgd2lkdGg6ICcxMDAlJ319PlxyXG4gICAgICAgICAgPENvbCBzdHlsZT17eyBkaXNwbGF5OiAnZmxleCcsIGFsaWduSXRlbXM6ICdjZW50ZXInLGZsb2F0OiAnbGVmdCcsIHBhZGRpbmc6IDggfX0+XHJcbiAgICAgICAgICAgIDxSdW5JbmZvIHF1ZXJ5PXtxdWVyeX0gLz5cclxuICAgICAgICAgICAgPENvbXBvc2VkU2VhcmNoIC8+XHJcbiAgICAgICAgICA8L0NvbD5cclxuICAgICAgICAgICB7LyogPENvbCBzdHlsZT17e2Rpc3BsYXk6ICdmbGV4JywgcG9zaXRpb246ICdhYnNvbHV0ZScsIHJpZ2h0OiAwfX0+XHJcbiAgICAgICAgICAgPFdvcmtzcGFjZXMgLz5cclxuICAgICAgICAgICA8UGxvdFNlYXJjaCBpc0xvYWRpbmdGb2xkZXJzPXtmYWxzZX0gLz5cclxuICAgICAgICAgPC9Db2w+ICovfVxyXG4gICAgICAgICA8L0NvbD5cclxuICAgICAgICApIDogKFxyXG4gICAgICAgICAgICA8PlxyXG4gICAgICAgICAgICAgIDxOYXZcclxuICAgICAgICAgICAgICAgIGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXI9e3F1ZXJ5LnNlYXJjaF9ydW5fbnVtYmVyfVxyXG4gICAgICAgICAgICAgICAgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lPXtxdWVyeS5zZWFyY2hfZGF0YXNldF9uYW1lfVxyXG4gICAgICAgICAgICAgICAgaGFuZGxlcj17bmF2aWdhdGlvbkhhbmRsZXJ9XHJcbiAgICAgICAgICAgICAgICB0eXBlPVwidG9wXCJcclxuICAgICAgICAgICAgICAvPlxyXG4gICAgICAgICAgICA8Lz5cclxuICAgICAgICAgIClcclxuICAgICAgfVxyXG4gICAgPC9Db2w+XHJcbiAgKTtcclxufTtcclxuIl0sInNvdXJjZVJvb3QiOiIifQ==