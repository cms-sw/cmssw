webpackHotUpdate_N_E("pages/index",{

/***/ "./components/overlayWithAnotherPlot/index.tsx":
/*!*****************************************************!*\
  !*** ./components/overlayWithAnotherPlot/index.tsx ***!
  \*****************************************************/
/*! exports provided: OverlayWithAnotherPlot */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "OverlayWithAnotherPlot", function() { return OverlayWithAnotherPlot; });
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! antd/lib/modal/Modal */ "./node_modules/antd/lib/modal/Modal.js");
/* harmony import */ var antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! next/router */ "./node_modules/next/dist/client/router.js");
/* harmony import */ var next_router__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(next_router__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var _containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../../containers/display/styledComponents */ "./containers/display/styledComponents.tsx");
/* harmony import */ var _containers_display_utils__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ../../containers/display/utils */ "./containers/display/utils.ts");
/* harmony import */ var _contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ../../contexts/leftSideContext */ "./contexts/leftSideContext.tsx");
/* harmony import */ var _hooks_useRequest__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../../hooks/useRequest */ "./hooks/useRequest.tsx");
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _containers_display_content_folderPath__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(/*! ../../containers/display/content/folderPath */ "./containers/display/content/folderPath.tsx");


var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/overlayWithAnotherPlot/index.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_1__["createElement"];









var OverlayWithAnotherPlot = function OverlayWithAnotherPlot(_ref) {
  _s();

  var visible = _ref.visible,
      setOpenOverlayWithAnotherPlotModal = _ref.setOpenOverlayWithAnotherPlotModal;

  var _React$useState = react__WEBPACK_IMPORTED_MODULE_1__["useState"]({
    folder_path: '',
    name: ''
  }),
      _React$useState2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState, 2),
      overlaidPlots = _React$useState2[0],
      setOverlaidPlots = _React$useState2[1];

  var _React$useState3 = react__WEBPACK_IMPORTED_MODULE_1__["useState"]([]),
      _React$useState4 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState3, 2),
      folderPath = _React$useState4[0],
      setFolderPath = _React$useState4[1];

  var _React$useState5 = react__WEBPACK_IMPORTED_MODULE_1__["useState"](''),
      _React$useState6 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState5, 2),
      currentFolder = _React$useState6[0],
      setCurrentFolder = _React$useState6[1];

  var _React$useState7 = react__WEBPACK_IMPORTED_MODULE_1__["useState"]({}),
      _React$useState8 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_0__["default"])(_React$useState7, 2),
      plot = _React$useState8[0],
      setPlot = _React$useState8[1];

  var router = Object(next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"])();
  var query = router.query;

  var _React$useContext = react__WEBPACK_IMPORTED_MODULE_1__["useContext"](_contexts_leftSideContext__WEBPACK_IMPORTED_MODULE_6__["store"]),
      updated_by_not_older_than = _React$useContext.updated_by_not_older_than;

  var params = {
    dataset_name: query.dataset_name,
    run_number: query.run_number,
    notOlderThan: updated_by_not_older_than,
    folders_path: overlaidPlots.folder_path,
    plot_name: overlaidPlots.name
  };
  var api = Object(_containers_display_utils__WEBPACK_IMPORTED_MODULE_5__["choose_api"])(params);
  var data_get_by_mount = Object(_hooks_useRequest__WEBPACK_IMPORTED_MODULE_7__["useRequest"])(api, {}, [overlaidPlots.folder_path]);
  var data = data_get_by_mount.data;
  var folders_or_plots = data ? data.data : [];

  var changeFolderPathByBreadcrumb = function changeFolderPathByBreadcrumb() {};

  console.log(currentFolder);
  return __jsx(antd_lib_modal_Modal__WEBPACK_IMPORTED_MODULE_2___default.a, {
    visible: visible,
    onCancel: function onCancel() {
      setOpenOverlayWithAnotherPlotModal(false);
      setFolderPath([]);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 48,
      columnNumber: 5
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_8__["Row"], {
    gutter: 16,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 55,
      columnNumber: 7
    }
  }, __jsx(antd__WEBPACK_IMPORTED_MODULE_8__["Col"], {
    style: {
      padding: 8
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 56,
      columnNumber: 9
    }
  }, __jsx(_containers_display_content_folderPath__WEBPACK_IMPORTED_MODULE_9__["FolderPath"], {
    folder_path: overlaidPlots.folder_path,
    changeFolderPathByBreadcrumb: changeFolderPathByBreadcrumb,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 57,
      columnNumber: 11
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_8__["Row"], {
    style: {
      width: '100%'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 59,
      columnNumber: 9
    }
  }, folders_or_plots.map(function (folder_or_plot) {
    return __jsx(react__WEBPACK_IMPORTED_MODULE_1__["Fragment"], null, folder_or_plot.subdir && __jsx(antd__WEBPACK_IMPORTED_MODULE_8__["Col"], {
      span: 8,
      onClick: function onClick() {
        return setCurrentFolder(folder_or_plot.subdir);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 65,
        columnNumber: 21
      }
    }, __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_4__["Icon"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 66,
        columnNumber: 23
      }
    }), __jsx(_containers_display_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledA"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 67,
        columnNumber: 23
      }
    }, folder_or_plot.subdir)));
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_8__["Row"], {
    style: {
      width: '100%'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 75,
      columnNumber: 9
    }
  }, folders_or_plots.map(function (folder_or_plot) {
    return __jsx(react__WEBPACK_IMPORTED_MODULE_1__["Fragment"], null, folder_or_plot.name && __jsx(antd__WEBPACK_IMPORTED_MODULE_8__["Col"], {
      span: 8,
      onClick: function onClick() {
        return setPlot(folder_or_plot);
      },
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 81,
        columnNumber: 21
      }
    }, __jsx(antd__WEBPACK_IMPORTED_MODULE_8__["Button"], {
      __self: _this,
      __source: {
        fileName: _jsxFileName,
        lineNumber: 82,
        columnNumber: 23
      }
    }, folder_or_plot.name)));
  }))));
};

_s(OverlayWithAnotherPlot, "jx1jYibJkFuVyGjoJX65lGHfbr0=", false, function () {
  return [next_router__WEBPACK_IMPORTED_MODULE_3__["useRouter"], _hooks_useRequest__WEBPACK_IMPORTED_MODULE_7__["useRequest"]];
});

_c = OverlayWithAnotherPlot;

var _c;

$RefreshReg$(_c, "OverlayWithAnotherPlot");

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

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9vdmVybGF5V2l0aEFub3RoZXJQbG90L2luZGV4LnRzeCJdLCJuYW1lcyI6WyJPdmVybGF5V2l0aEFub3RoZXJQbG90IiwidmlzaWJsZSIsInNldE9wZW5PdmVybGF5V2l0aEFub3RoZXJQbG90TW9kYWwiLCJSZWFjdCIsImZvbGRlcl9wYXRoIiwibmFtZSIsIm92ZXJsYWlkUGxvdHMiLCJzZXRPdmVybGFpZFBsb3RzIiwiZm9sZGVyUGF0aCIsInNldEZvbGRlclBhdGgiLCJjdXJyZW50Rm9sZGVyIiwic2V0Q3VycmVudEZvbGRlciIsInBsb3QiLCJzZXRQbG90Iiwicm91dGVyIiwidXNlUm91dGVyIiwicXVlcnkiLCJzdG9yZSIsInVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4iLCJwYXJhbXMiLCJkYXRhc2V0X25hbWUiLCJydW5fbnVtYmVyIiwibm90T2xkZXJUaGFuIiwiZm9sZGVyc19wYXRoIiwicGxvdF9uYW1lIiwiYXBpIiwiY2hvb3NlX2FwaSIsImRhdGFfZ2V0X2J5X21vdW50IiwidXNlUmVxdWVzdCIsImRhdGEiLCJmb2xkZXJzX29yX3Bsb3RzIiwiY2hhbmdlRm9sZGVyUGF0aEJ5QnJlYWRjcnVtYiIsImNvbnNvbGUiLCJsb2ciLCJwYWRkaW5nIiwid2lkdGgiLCJtYXAiLCJmb2xkZXJfb3JfcGxvdCIsInN1YmRpciJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBQ0E7QUFHQTtBQUNBO0FBQ0E7QUFDQTtBQUNBO0FBQ0E7QUFRTyxJQUFNQSxzQkFBc0IsR0FBRyxTQUF6QkEsc0JBQXlCLE9BQWtGO0FBQUE7O0FBQUEsTUFBL0VDLE9BQStFLFFBQS9FQSxPQUErRTtBQUFBLE1BQXRFQyxrQ0FBc0UsUUFBdEVBLGtDQUFzRTs7QUFBQSx3QkFDNUVDLDhDQUFBLENBQTRDO0FBQUVDLGVBQVcsRUFBRSxFQUFmO0FBQW1CQyxRQUFJLEVBQUU7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QUFBQTtBQUFBLE1BREYsRUFFRSxNQUFDLDRFQUFEO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsT0FBVUQsY0FBYyxDQUFDQyxNQUF6QixDQUZGLENBRkosQ0FERjtBQVVELEdBWEQsQ0FGSixDQUpGLEVBb0JFLE1BQUMsd0NBQUQ7QUFBSyxTQUFLLEVBQUU7QUFBRUgsV0FBSyxFQUFFO0FBQVQsS0FBWjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBRUlMLGdCQUFnQixDQUFDTSxHQUFqQixDQUFxQixVQUFDQyxjQUFELEVBQXlCO0FBQzVDLFdBQ0UsNERBQ0dBLGNBQWMsQ0FBQ2hDLElBQWYsSUFDQyxNQUFDLHdDQUFEO0FBQUssVUFBSSxFQUFFLENBQVg7QUFBYyxhQUFPLEVBQUU7QUFBQSxlQUFNUSxPQUFPLENBQUN3QixjQUFELENBQWI7QUFBQSxPQUF2QjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQ0UsTUFBQywyQ0FBRDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE9BQVVBLGNBQWMsQ0FBQ2hDLElBQXpCLENBREYsQ0FGSixDQURGO0FBU0QsR0FWRCxDQUZKLENBcEJGLENBUEYsQ0FERjtBQThDRCxDQTFFTTs7R0FBTUwsc0I7VUFNSWUscUQsRUFhV2EsNEQ7OztLQW5CZjVCLHNCIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LmM1YjMyMmVkOTgwODc5MzgzYmUwLmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgKiBhcyBSZWFjdCBmcm9tICdyZWFjdCdcclxuaW1wb3J0IE1vZGFsIGZyb20gJ2FudGQvbGliL21vZGFsL01vZGFsJ1xyXG5pbXBvcnQgeyB1c2VSb3V0ZXIgfSBmcm9tICduZXh0L3JvdXRlcidcclxuXHJcbmltcG9ydCB7IFBhcmFtc0ZvckFwaVByb3BzLCBQbG90b3ZlcmxhaWRTZXBhcmF0ZWx5UHJvcHMsIFF1ZXJ5UHJvcHMgfSBmcm9tICcuLi8uLi9jb250YWluZXJzL2Rpc3BsYXkvaW50ZXJmYWNlcydcclxuaW1wb3J0IHsgSWNvbiwgU3R5bGVkQSB9IGZyb20gJy4uLy4uL2NvbnRhaW5lcnMvZGlzcGxheS9zdHlsZWRDb21wb25lbnRzJ1xyXG5pbXBvcnQgeyBjaG9vc2VfYXBpIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L3V0aWxzJ1xyXG5pbXBvcnQgeyBzdG9yZSB9IGZyb20gJy4uLy4uL2NvbnRleHRzL2xlZnRTaWRlQ29udGV4dCdcclxuaW1wb3J0IHsgdXNlUmVxdWVzdCB9IGZyb20gJy4uLy4uL2hvb2tzL3VzZVJlcXVlc3QnXHJcbmltcG9ydCB7IEJ1dHRvbiwgQ29sLCBSb3cgfSBmcm9tICdhbnRkJ1xyXG5pbXBvcnQgeyBGb2xkZXJQYXRoIH0gZnJvbSAnLi4vLi4vY29udGFpbmVycy9kaXNwbGF5L2NvbnRlbnQvZm9sZGVyUGF0aCdcclxuaW1wb3J0IHsgUGFyc2VkVXJsUXVlcnlJbnB1dCB9IGZyb20gJ3F1ZXJ5c3RyaW5nJ1xyXG5cclxuaW50ZXJmYWNlIE92ZXJsYXlXaXRoQW5vdGhlclBsb3RQcm9wcyB7XHJcbiAgdmlzaWJsZTogYm9vbGVhbjtcclxuICBzZXRPcGVuT3ZlcmxheVdpdGhBbm90aGVyUGxvdE1vZGFsOiBhbnlcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IE92ZXJsYXlXaXRoQW5vdGhlclBsb3QgPSAoeyB2aXNpYmxlLCBzZXRPcGVuT3ZlcmxheVdpdGhBbm90aGVyUGxvdE1vZGFsIH06IE92ZXJsYXlXaXRoQW5vdGhlclBsb3RQcm9wcykgPT4ge1xyXG4gIGNvbnN0IFtvdmVybGFpZFBsb3RzLCBzZXRPdmVybGFpZFBsb3RzXSA9IFJlYWN0LnVzZVN0YXRlPFBsb3RvdmVybGFpZFNlcGFyYXRlbHlQcm9wcz4oeyBmb2xkZXJfcGF0aDogJycsIG5hbWU6ICcnIH0pXHJcbiAgY29uc3QgW2ZvbGRlclBhdGgsIHNldEZvbGRlclBhdGhdID0gUmVhY3QudXNlU3RhdGU8c3RyaW5nW10+KFtdKVxyXG4gIGNvbnN0IFtjdXJyZW50Rm9sZGVyLCBzZXRDdXJyZW50Rm9sZGVyXSA9IFJlYWN0LnVzZVN0YXRlKCcnKVxyXG4gIGNvbnN0IFtwbG90LCBzZXRQbG90XSA9IFJlYWN0LnVzZVN0YXRlKHt9KVxyXG5cclxuICBjb25zdCByb3V0ZXIgPSB1c2VSb3V0ZXIoKTtcclxuICBjb25zdCBxdWVyeTogUXVlcnlQcm9wcyA9IHJvdXRlci5xdWVyeTtcclxuICBjb25zdCB7IHVwZGF0ZWRfYnlfbm90X29sZGVyX3RoYW4gfSA9IFJlYWN0LnVzZUNvbnRleHQoc3RvcmUpXHJcblxyXG4gIGNvbnN0IHBhcmFtczogUGFyYW1zRm9yQXBpUHJvcHMgPSB7XHJcbiAgICBkYXRhc2V0X25hbWU6IHF1ZXJ5LmRhdGFzZXRfbmFtZSBhcyBzdHJpbmcsXHJcbiAgICBydW5fbnVtYmVyOiBxdWVyeS5ydW5fbnVtYmVyIGFzIHN0cmluZyxcclxuICAgIG5vdE9sZGVyVGhhbjogdXBkYXRlZF9ieV9ub3Rfb2xkZXJfdGhhbixcclxuICAgIGZvbGRlcnNfcGF0aDogb3ZlcmxhaWRQbG90cy5mb2xkZXJfcGF0aCxcclxuICAgIHBsb3RfbmFtZTogb3ZlcmxhaWRQbG90cy5uYW1lXHJcbiAgfVxyXG5cclxuICBjb25zdCBhcGkgPSBjaG9vc2VfYXBpKHBhcmFtcylcclxuICBjb25zdCBkYXRhX2dldF9ieV9tb3VudCA9IHVzZVJlcXVlc3QoYXBpLFxyXG4gICAge30sXHJcbiAgICBbb3ZlcmxhaWRQbG90cy5mb2xkZXJfcGF0aF1cclxuICApO1xyXG4gIGNvbnN0IHsgZGF0YSB9ID0gZGF0YV9nZXRfYnlfbW91bnRcclxuICBjb25zdCBmb2xkZXJzX29yX3Bsb3RzID0gZGF0YSA/IGRhdGEuZGF0YSA6IFtdXHJcbiAgY29uc3QgY2hhbmdlRm9sZGVyUGF0aEJ5QnJlYWRjcnVtYiA9ICgpID0+IHsgfVxyXG5cclxuICBjb25zb2xlLmxvZyhjdXJyZW50Rm9sZGVyKVxyXG4gIHJldHVybiAoXHJcbiAgICA8TW9kYWxcclxuICAgICAgdmlzaWJsZT17dmlzaWJsZX1cclxuICAgICAgb25DYW5jZWw9eygpID0+IHtcclxuICAgICAgICBzZXRPcGVuT3ZlcmxheVdpdGhBbm90aGVyUGxvdE1vZGFsKGZhbHNlKVxyXG4gICAgICAgIHNldEZvbGRlclBhdGgoW10pXHJcbiAgICAgIH19XHJcbiAgICA+XHJcbiAgICAgIDxSb3cgZ3V0dGVyPXsxNn0+XHJcbiAgICAgICAgPENvbCBzdHlsZT17eyBwYWRkaW5nOiA4IH19PlxyXG4gICAgICAgICAgPEZvbGRlclBhdGggZm9sZGVyX3BhdGg9e292ZXJsYWlkUGxvdHMuZm9sZGVyX3BhdGh9IGNoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWI9e2NoYW5nZUZvbGRlclBhdGhCeUJyZWFkY3J1bWJ9IC8+XHJcbiAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgPFJvdyBzdHlsZT17eyB3aWR0aDogJzEwMCUnIH19PlxyXG4gICAgICAgICAge1xyXG4gICAgICAgICAgICBmb2xkZXJzX29yX3Bsb3RzLm1hcCgoZm9sZGVyX29yX3Bsb3Q6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgICAgICA8PlxyXG4gICAgICAgICAgICAgICAgICB7Zm9sZGVyX29yX3Bsb3Quc3ViZGlyICYmXHJcbiAgICAgICAgICAgICAgICAgICAgPENvbCBzcGFuPXs4fSBvbkNsaWNrPXsoKSA9PiBzZXRDdXJyZW50Rm9sZGVyKGZvbGRlcl9vcl9wbG90LnN1YmRpcil9PlxyXG4gICAgICAgICAgICAgICAgICAgICAgPEljb24gLz5cclxuICAgICAgICAgICAgICAgICAgICAgIDxTdHlsZWRBPntmb2xkZXJfb3JfcGxvdC5zdWJkaXJ9PC9TdHlsZWRBPlxyXG4gICAgICAgICAgICAgICAgICAgIDwvQ29sPlxyXG4gICAgICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgICA8Lz5cclxuICAgICAgICAgICAgICApXHJcbiAgICAgICAgICAgIH0pXHJcbiAgICAgICAgICB9XHJcbiAgICAgICAgPC9Sb3c+XHJcbiAgICAgICAgPFJvdyBzdHlsZT17eyB3aWR0aDogJzEwMCUnIH19PlxyXG4gICAgICAgICAge1xyXG4gICAgICAgICAgICBmb2xkZXJzX29yX3Bsb3RzLm1hcCgoZm9sZGVyX29yX3Bsb3Q6IGFueSkgPT4ge1xyXG4gICAgICAgICAgICAgIHJldHVybiAoXHJcbiAgICAgICAgICAgICAgICA8PlxyXG4gICAgICAgICAgICAgICAgICB7Zm9sZGVyX29yX3Bsb3QubmFtZSAmJlxyXG4gICAgICAgICAgICAgICAgICAgIDxDb2wgc3Bhbj17OH0gb25DbGljaz17KCkgPT4gc2V0UGxvdChmb2xkZXJfb3JfcGxvdCl9PlxyXG4gICAgICAgICAgICAgICAgICAgICAgPEJ1dHRvbiA+e2ZvbGRlcl9vcl9wbG90Lm5hbWV9PC9CdXR0b24+XHJcbiAgICAgICAgICAgICAgICAgICAgPC9Db2w+XHJcbiAgICAgICAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgICAgIDwvPlxyXG4gICAgICAgICAgICAgIClcclxuICAgICAgICAgICAgfSlcclxuICAgICAgICAgIH1cclxuICAgICAgICA8L1Jvdz5cclxuICAgICAgPC9Sb3c+XHJcbiAgICA8L01vZGFsPlxyXG4gIClcclxufSJdLCJzb3VyY2VSb290IjoiIn0=